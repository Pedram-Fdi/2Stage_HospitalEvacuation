import re
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "plots.py"
text = p.read_text(encoding="utf-8")
text2 = re.sub(
    r'_savefig\(fig,\s*plot_dir\s*/\s*("[^"]+\.png")\)',
    r"_savefig(fig, _out(plot_dir, \1, instance_name))",
    text,
)
text2 = text2.replace(
    "def generate_factorial_plots(df: pd.DataFrame, plot_dir: Path) -> None:",
    "def generate_factorial_plots(df: pd.DataFrame, plot_dir: Path, instance_name: Optional[str] = None) -> None:",
)
text2 = text2.replace(
    "def generate_factorial_baseline_comparison_plots(df: pd.DataFrame, plot_dir: Path) -> None:",
    "def generate_factorial_baseline_comparison_plots(df: pd.DataFrame, plot_dir: Path, instance_name: Optional[str] = None) -> None:",
)
print("old", text.count("_savefig(fig, plot_dir /"))
print("new", text2.count("_savefig(fig, _out(plot_dir,"))
p.write_text(text2, encoding="utf-8")
