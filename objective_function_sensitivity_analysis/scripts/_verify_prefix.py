import argparse
import inspect
from pathlib import Path

from objective_function_sensitivity_analysis.plots import _prefixed_filename
from objective_function_sensitivity_analysis import plots
from objective_function_sensitivity_analysis.runner import (
    artifact_path,
    select_weight_schemes,
)

assert _prefixed_filename("foo.png", "4_20_5_15_3_3_CRP") == "4_20_5_15_3_3_CRP_foo.png"
assert (
    _prefixed_filename("4_20_5_15_3_3_CRP_foo.png", "4_20_5_15_3_3_CRP")
    == "4_20_5_15_3_3_CRP_foo.png"
)

args = argparse.Namespace(
    SensitivityMode="factorial",
    ObjectiveWeights=None,
    RunAllSchemes=False,
    OATMultipliers="auto",
    FactorialLevels="auto",
)
schemes = select_weight_schemes(args)
assert "Baseline" in schemes
assert sum(1 for k in schemes if k.startswith("FAC_")) == 27
print("schemes", len(schemes), "ok")
print(
    "artifact",
    artifact_path(Path("results"), "acf_spatial_geometry_by_scheme.csv", "4_20_5_15_3_3_CRP"),
)
assert "instance_name" in inspect.signature(plots.generate_factorial_plots).parameters
assert "instance_name" in inspect.signature(plots.generate_all_plots).parameters
src = Path(__file__).resolve().parents[1] / "plots.py"
text = src.read_text(encoding="utf-8")
print("leftover plot_dir slash png:", text.count('plot_dir / "'))
print("verify ok")
