"""Explore configured GDP-model parameter extremes for any APEC economy.

Purpose and interpretation
--------------------------
This analysis estimates the range of an economy's real GDP over a selected
horizon when the current APERC Solow-Swan model's configured parameter limits
and bundled UN DESA population variants are stressed together. It deliberately
calls the repository's authoritative ``aperc_gdp_model`` implementation rather
than duplicating the model equations.

The result is a model stress-test envelope, not a forecast confidence interval.
Parameter limits are the minimum and maximum values already used for at least
one economy in ``config/gdp_model_parameters.csv``. Some limits—especially the
upper labour-efficiency corridor—may have been calibrated for a different
economy and should not automatically be interpreted as plausible for the
selected economy.

Method
------
1. Select an economy code, reporting period, milestone years, and output prefix.
   Historical hand-offs remain authoritative: Chinese Taipei (``18_CT``) uses
   IMF data through 2031; the other economies use WDI data through 2025. The
   underlying model continues calculating through its fixed final year, 2100.
2. Read the selected economy's calibrated parameters and derive observed
   minima/maxima from the complete authoritative parameter table.
3. Move the efficiency, savings, and depreciation corridor endpoints as pairs.
   This prevents invalid mixed corners where a lower endpoint exceeds its upper
   endpoint.
4. Use both observed endpoints for every other varying parameter. A parameter
   with one configured value, currently ``alpha``, remains fixed and is reported
   as such.
5. Evaluate the complete two-level factorial across the varying axes for Low,
   Medium, and High population paths. The current table produces 256 parameter
   corners times three population paths, or 768 stress cases.
6. Run the selected economy's calibrated parameters against every population
   path, plus one-axis-at-a-time diagnostics around the calibrated Medium case.
7. Construct an annual envelope across stress corners and calibrated cases,
   report milestone GDP, and retain the exact cases attaining the horizon's
   endpoints. Calibrated cases are included because this nonlinear model can
   briefly place an interior calibration outside a pure-corner envelope.
8. Validate unique case/year rows, complete annual coverage, finite positive GDP,
   the expected number of stress cases, and a common GDP value in the start year.

Outputs
-------
``run_analysis`` returns the main dataframes and writes the following files to
the selected output directory:

* ``<prefix>_parameter_bounds.csv`` — calibration and observed limits.
* ``<prefix>_corner_summary.csv`` — one row per case with milestone GDP.
* ``<prefix>_all_paths.csv`` — annual GDP for every stress/calibrated case.
* ``<prefix>_envelope_by_year.csv`` — annual minimum, calibrated, and maximum.
* ``<prefix>_extreme_parameters.csv`` — settings attaining horizon endpoints.
* ``<prefix>_one_at_a_time.csv`` — isolated axis effects at the horizon.
* ``<prefix>_validation.txt`` — reproducibility and integrity checks.

Command-line example
--------------------
Run from the repository root with the project interpreter::

    /home/dev/Documents/uvenvs/st/bin/python scripts/gdp_parameter_extremes.py \
        --economy-code 15_PHL --end-year 2070 --output-prefix philippines

Google Colab
------------
The bundled notebooks shallow-clone the repository with ``git clone --depth 1``
and import this module from ``scripts/``. Existing Colab clones update with
``git pull --ff-only``. Consequently, this script and the notebooks must be
committed and pushed before an uploaded notebook can use them from GitHub.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = DEFAULT_PROJECT_ROOT / "notebooks" / "outputs"

if str(DEFAULT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_PROJECT_ROOT))

from core import postprocess  # noqa: E402
from macro_model.model import (  # noqa: E402
    FINAL_YEAR,
    aperc_gdp_model,
    jump_off_year_for_economy,
)
from macro_model.prepare_model_inputs import (  # noqa: E402
    build_gdp_input_table,
    build_labour_efficiency,
)


POPULATION_FILES = {
    "Low": "undesa_pop_to2100_low.csv",
    "Medium": "undesa_pop_to2100_med.csv",
    "High": "undesa_pop_to2100_high.csv",
}

# Corridor endpoints must move together. Independently mixing their global
# extrema can create nonsensical cases where low > high.
CORRIDOR_AXES = {
    "eff_corridor": ("high_eff", "low_eff"),
    "savings_corridor": ("high_sav", "low_sav"),
    "depreciation_corridor": ("high_delta", "low_delta"),
}
SCALAR_PARAMETERS = (
    "lab_eff_periods",
    "change_eff",
    "change_sav",
    "change_del",
    "alpha",
    "cap_compare",
)


@dataclass(frozen=True)
class AxisEndpoint:
    """One endpoint of a scalar parameter or coupled corridor axis."""

    axis: str
    endpoint: str
    updates: dict[str, float | int]


def load_parameter_space(
    economy_code: str,
    config_path: Path,
) -> tuple[dict[str, float | int], list[AxisEndpoint], pd.DataFrame]:
    """Return one economy's calibration, search endpoints, and bounds table."""
    parameter_table = pd.read_csv(config_path)
    if economy_code not in set(parameter_table["economy_code"]):
        available = ", ".join(sorted(parameter_table["economy_code"]))
        raise ValueError(f"Unknown economy code {economy_code!r}. Available codes: {available}")
    parameter_columns = [
        column
        for column in parameter_table.columns
        if column not in {"economy_code", "notes"}
    ]
    calibrated = (
        parameter_table.set_index("economy_code").loc[economy_code, parameter_columns].to_dict()
    )
    calibrated["lab_eff_periods"] = int(calibrated["lab_eff_periods"])

    bounds_rows = []
    for parameter in parameter_columns:
        minimum = parameter_table[parameter].min()
        maximum = parameter_table[parameter].max()
        bounds_rows.append(
            {
                "parameter": parameter,
                "economy_calibrated": calibrated[parameter],
                "observed_min": minimum,
                "observed_max": maximum,
                "varies_in_search": bool(minimum != maximum),
            }
        )
    bounds = pd.DataFrame(bounds_rows)

    endpoints: list[AxisEndpoint] = []
    for axis, parameters in CORRIDOR_AXES.items():
        low_updates = {parameter: parameter_table[parameter].min() for parameter in parameters}
        high_updates = {parameter: parameter_table[parameter].max() for parameter in parameters}
        if low_updates != high_updates:
            endpoints.extend(
                [
                    AxisEndpoint(axis, "min", low_updates),
                    AxisEndpoint(axis, "max", high_updates),
                ]
            )

    for parameter in SCALAR_PARAMETERS:
        minimum = parameter_table[parameter].min()
        maximum = parameter_table[parameter].max()
        if minimum == maximum:
            continue
        if parameter == "lab_eff_periods":
            minimum, maximum = int(minimum), int(maximum)
        endpoints.extend(
            [
                AxisEndpoint(parameter, "min", {parameter: minimum}),
                AxisEndpoint(parameter, "max", {parameter: maximum}),
            ]
        )

    return calibrated, endpoints, bounds


def group_endpoints(endpoints: list[AxisEndpoint]) -> dict[str, list[AxisEndpoint]]:
    """Group the two endpoints belonging to each search axis."""
    grouped: dict[str, list[AxisEndpoint]] = {}
    for endpoint in endpoints:
        grouped.setdefault(endpoint.axis, []).append(endpoint)
    for axis, values in grouped.items():
        if {value.endpoint for value in values} != {"min", "max"}:
            raise ValueError(f"Axis {axis!r} does not have min/max endpoints")
        grouped[axis] = sorted(values, key=lambda value: value.endpoint)
    return grouped


def enumerate_parameter_corners(
    calibrated: dict[str, float | int], endpoints: list[AxisEndpoint]
):
    """Yield every coherent two-level parameter corner."""
    grouped = group_endpoints(endpoints)
    axes = list(grouped)
    choices = [grouped[axis] for axis in axes]
    for corner_number, selected in enumerate(itertools.product(*choices), start=1):
        params = calibrated.copy()
        labels = {}
        for endpoint in selected:
            params.update(endpoint.updates)
            labels[endpoint.axis] = endpoint.endpoint
        yield corner_number, labels, params


class EconomyRunner:
    """Load pipeline data once and run repeatable cases for one economy."""

    def __init__(self, project_root: Path, economy_code: str, start_year: int, end_year: int) -> None:
        results_data_dir = project_root / "results" / "data"
        self.economy_code = economy_code
        self.start_year = start_year
        self.end_year = end_year
        self.imf = pd.read_csv(results_data_dir / "IMF_to2031.csv")
        self.wdi = pd.read_csv(results_data_dir / "WDI_to2025.csv")
        self.capital = pd.read_csv(results_data_dir / "capital_stock.csv")
        self.delta = pd.read_csv(results_data_dir / "PWT_delta_2023.csv")
        self.imf_savings = pd.read_csv(results_data_dir / "IMF_savings_2031.csv")
        self.wdi_savings = pd.read_csv(results_data_dir / "WDI_savings_2025.csv")
        self.pwt = pd.read_csv(results_data_dir / "PWT_cap_labour_to2023.csv")
        self.gdp_9th = pd.read_csv(results_data_dir / "GDP_9th.csv")
        self.populations = {
            scenario: pd.read_csv(results_data_dir / filename)
            for scenario, filename in POPULATION_FILES.items()
        }
        self.savings = pd.concat(
            [
                self.imf_savings[self.imf_savings["economy_code"] == "18_CT"],
                self.wdi_savings[
                    (self.wdi_savings["economy_code"] != "18_CT")
                    & (self.wdi_savings["year"] == 2025)
                ],
            ],
            ignore_index=True,
        )
        self.capital_growth = self.capital[
            ["economy_code", "variable", "year", "percent"]
        ].copy()
        self._prepared: dict[tuple[str, float], tuple[pd.DataFrame, pd.DataFrame]] = {}

    def prepare(self, population_scenario: str, alpha: float):
        """Prepare input and historical efficiency paths, cached by alpha."""
        key = (population_scenario, float(alpha))
        if key not in self._prepared:
            population = self.populations[population_scenario]
            labour_efficiency = build_labour_efficiency(
                self.imf,
                population,
                self.capital,
                [self.economy_code],
                wdi_long=self.wdi,
                alpha_by_economy={self.economy_code: alpha},
            )
            input_data = build_gdp_input_table(
                population,
                self.imf,
                labour_efficiency,
                self.capital,
                [self.economy_code],
                wdi_long=self.wdi,
            )
            self._prepared[key] = input_data, labour_efficiency
        return self._prepared[key]

    def run(self, population_scenario: str, params: dict[str, float | int]) -> pd.DataFrame:
        """Run one case and return its continuous annual GDP path."""
        input_data, labour_efficiency = self.prepare(population_scenario, params["alpha"])
        long_output, _ = aperc_gdp_model(
            economy=self.economy_code,
            input_data=input_data,
            labour_data=labour_efficiency,
            cap_growth_data=self.capital_growth,
            delta_data=self.delta,
            sav_data=self.savings,
            save_invest_hist=self.imf,
            delta_hist=self.pwt,
            gdp_9th=self.gdp_9th,
            wdi_savings_hist=self.wdi_savings,
            jump_off_year=jump_off_year_for_economy(self.economy_code),
            **params,
        )
        processed = postprocess(long_output)
        path = processed[
            (processed["variable"] == "real_GDP")
            & processed["year"].between(self.start_year, self.end_year)
        ][["year", "value"]].rename(columns={"value": "gdp_millions_2021_usd_ppp"})
        return path.reset_index(drop=True)


def case_path(
    runner: EconomyRunner,
    scenario_id: str,
    case_type: str,
    population_scenario: str,
    params: dict[str, float | int],
) -> pd.DataFrame:
    """Run and label one annual GDP path."""
    path = runner.run(population_scenario, params)
    path.insert(0, "economy_code", runner.economy_code)
    path.insert(0, "population_scenario", population_scenario)
    path.insert(0, "case_type", case_type)
    path.insert(0, "scenario_id", scenario_id)
    return path


def summarize_paths(
    paths: pd.DataFrame,
    parameters: pd.DataFrame,
    milestone_years: tuple[int, ...],
) -> pd.DataFrame:
    """Create one scenario row with GDP at each milestone."""
    milestones = (
        paths[paths["year"].isin(milestone_years)]
        .pivot(index="scenario_id", columns="year", values="gdp_millions_2021_usd_ppp")
        .rename(columns=lambda year: f"gdp_{year}_millions")
        .reset_index()
    )
    metadata = paths[
        ["scenario_id", "economy_code", "case_type", "population_scenario"]
    ].drop_duplicates("scenario_id")
    return metadata.merge(parameters, on="scenario_id", how="left").merge(
        milestones, on="scenario_id", how="left"
    )


def build_envelope(paths: pd.DataFrame) -> pd.DataFrame:
    """Return annual limits across all evaluated cases and calibrated Medium GDP."""
    # Include calibrated cases because this nonlinear model can briefly put an
    # interior calibration outside the envelope formed only by pure corners.
    low_indices = paths.groupby("year")["gdp_millions_2021_usd_ppp"].idxmin()
    high_indices = paths.groupby("year")["gdp_millions_2021_usd_ppp"].idxmax()
    low = paths.loc[low_indices, ["year", "scenario_id", "population_scenario", "gdp_millions_2021_usd_ppp"]].rename(
        columns={
            "scenario_id": "minimum_scenario_id",
            "population_scenario": "minimum_population_scenario",
            "gdp_millions_2021_usd_ppp": "minimum_gdp_millions",
        }
    )
    high = paths.loc[high_indices, ["year", "scenario_id", "population_scenario", "gdp_millions_2021_usd_ppp"]].rename(
        columns={
            "scenario_id": "maximum_scenario_id",
            "population_scenario": "maximum_population_scenario",
            "gdp_millions_2021_usd_ppp": "maximum_gdp_millions",
        }
    )
    baseline = paths[
        (paths["scenario_id"] == "calibrated__Medium")
    ][["year", "gdp_millions_2021_usd_ppp"]].rename(
        columns={"gdp_millions_2021_usd_ppp": "calibrated_medium_gdp_millions"}
    )
    envelope = low.merge(baseline, on="year").merge(high, on="year")
    envelope["range_gdp_millions"] = (
        envelope["maximum_gdp_millions"] - envelope["minimum_gdp_millions"]
    )
    return envelope


def build_one_at_a_time(
    runner: EconomyRunner,
    calibrated: dict[str, float | int],
    endpoints: list[AxisEndpoint],
    end_year: int,
) -> pd.DataFrame:
    """Measure each endpoint around calibrated parameters and Medium population."""
    base_at_horizon = runner.run("Medium", calibrated).set_index("year").loc[
        end_year, "gdp_millions_2021_usd_ppp"
    ]
    rows = [
        {
            "axis": "calibrated",
            "endpoint": "calibrated",
            "gdp_horizon_millions": base_at_horizon,
            "difference_from_calibrated_millions": 0.0,
            "percent_difference_from_calibrated": 0.0,
        }
    ]
    for endpoint in endpoints:
        params = calibrated.copy()
        params.update(endpoint.updates)
        gdp_at_horizon = runner.run("Medium", params).set_index("year").loc[
            end_year, "gdp_millions_2021_usd_ppp"
        ]
        rows.append(
            {
                "axis": endpoint.axis,
                "endpoint": endpoint.endpoint,
                "gdp_horizon_millions": gdp_at_horizon,
                "difference_from_calibrated_millions": gdp_at_horizon - base_at_horizon,
                "percent_difference_from_calibrated": (gdp_at_horizon / base_at_horizon - 1) * 100,
            }
        )
    return pd.DataFrame(rows)


def validate(
    paths: pd.DataFrame,
    expected_corner_cases: int,
    start_year: int,
    end_year: int,
    economy_code: str,
) -> list[str]:
    """Raise on invalid output and return human-readable validation lines."""
    expected_years = set(range(start_year, end_year + 1))
    duplicates = paths.duplicated(["scenario_id", "year"]).sum()
    if duplicates:
        raise ValueError(f"Found {duplicates} duplicate scenario/year rows")

    bad_year_cases = []
    for scenario_id, group in paths.groupby("scenario_id"):
        if set(group["year"]) != expected_years:
            bad_year_cases.append(scenario_id)
    if bad_year_cases:
        raise ValueError(f"Cases with missing years: {bad_year_cases[:5]}")

    values = paths["gdp_millions_2021_usd_ppp"]
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("GDP paths contain missing, infinite, or non-positive values")

    corner_count = paths.loc[paths["case_type"] == "corner", "scenario_id"].nunique()
    if corner_count != expected_corner_cases:
        raise ValueError(f"Expected {expected_corner_cases} corner cases, got {corner_count}")

    anchors = paths[paths["year"] == start_year]["gdp_millions_2021_usd_ppp"]
    if not np.allclose(anchors, anchors.iloc[0]):
        raise ValueError(f"Cases do not share one GDP anchor in {start_year}")

    return [
        "PASS: no duplicate scenario/year rows",
        f"PASS: every scenario covers each year from {start_year} through {end_year}",
        "PASS: every GDP value is finite and positive",
        f"PASS: {corner_count} parameter/population stress cases were evaluated",
        f"PASS: all {economy_code} cases share the {start_year} GDP anchor "
        f"({anchors.iloc[0]:,.3f} million 2021 USD PPP)",
    ]


OUTPUT_SUFFIXES = {
    "bounds": "parameter_bounds.csv",
    "summary": "corner_summary.csv",
    "paths": "all_paths.csv",
    "envelope": "envelope_by_year.csv",
    "extremes": "extreme_parameters.csv",
    "one_at_a_time": "one_at_a_time.csv",
    "validation": "validation.txt",
}


def output_paths(output_dir: Path, output_prefix: str) -> dict[str, Path]:
    """Return the output path for each analysis artifact."""
    if not output_prefix or Path(output_prefix).name != output_prefix:
        raise ValueError("output_prefix must be a non-empty file-name prefix")
    return {
        name: output_dir / f"{output_prefix}_{suffix}"
        for name, suffix in OUTPUT_SUFFIXES.items()
    }


def load_analysis_outputs(
    output_dir: Path,
    output_prefix: str,
) -> dict[str, pd.DataFrame]:
    """Load previously generated tabular outputs."""
    files = output_paths(Path(output_dir), output_prefix)
    return {
        name: pd.read_csv(path)
        for name, path in files.items()
        if name != "validation"
    }


def run_analysis(
    economy_code: str,
    end_year: int,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    start_year: int | None = None,
    milestone_years: tuple[int, ...] | None = None,
    output_prefix: str | None = None,
    project_root: Path = DEFAULT_PROJECT_ROOT,
) -> dict[str, pd.DataFrame]:
    """Run the full analysis for one economy, save CSVs, and return dataframes."""
    project_root = Path(project_root).resolve()
    output_dir = Path(output_dir).resolve()
    economy_code = str(economy_code)
    start_year = jump_off_year_for_economy(economy_code) if start_year is None else int(start_year)
    end_year = int(end_year)
    if end_year > FINAL_YEAR:
        raise ValueError(f"end_year cannot exceed the model's final year ({FINAL_YEAR})")
    if start_year > end_year:
        raise ValueError("start_year must not be later than end_year")
    if milestone_years is None:
        milestone_years = tuple(
            year for year in range(((start_year // 10) + 1) * 10, end_year + 1, 10)
        )
        if end_year not in milestone_years:
            milestone_years = (*milestone_years, end_year)
    milestone_years = tuple(
        sorted({int(year) for year in milestone_years if start_year <= int(year) <= end_year})
    )
    if not milestone_years:
        milestone_years = (end_year,)
    output_prefix = output_prefix or economy_code.lower()

    output_dir.mkdir(parents=True, exist_ok=True)
    files = output_paths(output_dir, output_prefix)
    calibrated, endpoints, bounds = load_parameter_space(
        economy_code,
        project_root / "config" / "gdp_model_parameters.csv",
    )
    grouped = group_endpoints(endpoints)
    corners = list(enumerate_parameter_corners(calibrated, endpoints))
    expected_corner_cases = len(corners) * len(POPULATION_FILES)
    runner = EconomyRunner(project_root, economy_code, start_year, end_year)

    paths = []
    parameter_rows = []
    for population_scenario in POPULATION_FILES:
        scenario_id = f"calibrated__{population_scenario}"
        paths.append(
            case_path(
                runner,
                scenario_id,
                "calibrated",
                population_scenario,
                calibrated,
            )
        )
        parameter_rows.append({"scenario_id": scenario_id, **calibrated})

    for corner_number, labels, params in corners:
        for population_scenario in POPULATION_FILES:
            scenario_id = f"corner_{corner_number:04d}__{population_scenario}"
            paths.append(
                case_path(
                    runner,
                    scenario_id,
                    "corner",
                    population_scenario,
                    params,
                )
            )
            parameter_rows.append(
                {
                    "scenario_id": scenario_id,
                    **{f"axis_{axis}": endpoint for axis, endpoint in labels.items()},
                    **params,
                }
            )

    all_paths = pd.concat(paths, ignore_index=True)
    parameters = pd.DataFrame(parameter_rows)
    validation_lines = validate(
        all_paths,
        expected_corner_cases,
        start_year,
        end_year,
        economy_code,
    )
    summary = summarize_paths(all_paths, parameters, milestone_years)
    envelope = build_envelope(all_paths)

    endpoint_ids = set(
        envelope.loc[envelope["year"] == end_year, ["minimum_scenario_id", "maximum_scenario_id"]]
        .iloc[0]
        .tolist()
    )
    extreme_parameters = summary[summary["scenario_id"].isin(endpoint_ids)].copy()
    extreme_parameters.insert(
        1,
        "extreme",
        extreme_parameters["scenario_id"].map(
            {
                envelope.loc[envelope["year"] == end_year, "minimum_scenario_id"].iloc[0]: "minimum",
                envelope.loc[envelope["year"] == end_year, "maximum_scenario_id"].iloc[0]: "maximum",
            }
        ),
    )
    one_at_a_time = build_one_at_a_time(runner, calibrated, endpoints, end_year)

    bounds.to_csv(files["bounds"], index=False)
    summary.to_csv(files["summary"], index=False)
    all_paths.to_csv(files["paths"], index=False)
    envelope.to_csv(files["envelope"], index=False)
    extreme_parameters.to_csv(files["extremes"], index=False)
    one_at_a_time.to_csv(files["one_at_a_time"], index=False)
    files["validation"].write_text(
        "\n".join(validation_lines) + "\n", encoding="utf-8"
    )

    final = envelope[envelope["year"] == end_year].iloc[0]
    print(f"Evaluated {expected_corner_cases:,} stress cases across {len(grouped)} axes.")
    print(
        f"{economy_code} {end_year} GDP stress range: "
        f"{final['minimum_gdp_millions']:,.0f} to "
        f"{final['maximum_gdp_millions']:,.0f} million 2021 USD PPP"
    )
    print(
        f"Calibrated Medium-population GDP: "
        f"{final['calibrated_medium_gdp_millions']:,.0f} million 2021 USD PPP"
    )
    print(f"Outputs written to {output_dir}")

    return {
        "bounds": bounds,
        "summary": summary,
        "paths": all_paths,
        "envelope": envelope,
        "extremes": extreme_parameters,
        "one_at_a_time": one_at_a_time,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--economy-code", required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--milestone-years", type=int, nargs="*")
    parser.add_argument("--output-prefix")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV outputs (default: notebooks/outputs)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_analysis(
        economy_code=arguments.economy_code,
        start_year=arguments.start_year,
        end_year=arguments.end_year,
        milestone_years=(
            tuple(arguments.milestone_years) if arguments.milestone_years else None
        ),
        output_prefix=arguments.output_prefix,
        output_dir=arguments.output_dir,
    )
