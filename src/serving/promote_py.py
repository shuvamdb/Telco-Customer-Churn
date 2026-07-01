from pathlib import Path
import shutil


experiment_id = "860777467256719846"
run_id = "911aacb584b545bb850df55e1dc5244a"


source = (
    Path("mlruns")
    / experiment_id
    / run_id
)


destination = (
    Path("src")
    / "serving"
    / "model"
    / run_id
)


if not source.exists():
    raise FileNotFoundError(
        f"MLflow run not found: {source}"
    )


destination.mkdir(
    parents=True,
    exist_ok=True
)


shutil.copytree(
    source,
    destination,
    dirs_exist_ok=True
)


print("MLflow run copied successfully")
print(f"Copied from: {source}")
print(f"Copied to: {destination}")