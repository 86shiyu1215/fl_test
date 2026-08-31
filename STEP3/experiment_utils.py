from datetime import datetime
from pathlib import Path
import csv
import shutil


# ============================================================
# 保存対象となるコード
# ============================================================

CODE_FILES = [
    "model.py",
    "task.py",
    "client_app.py",
    "server_app.py",
    "pyproject.toml",
]


# ============================================================
# 実験フォルダを作成
# ============================================================

def create_experiment_directory(
    base_dir,
    experiment_id,
):
    """
    例:
    experiments/
        exp_002_zscore_baseline/
            run_20260831_171500/
    """

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    experiment_dir = (
        Path(base_dir)
        / "experiments"
        / experiment_id
        / f"run_{timestamp}"
    )

    result_dir = (
        experiment_dir
        / "results"
    )

    code_dir = (
        experiment_dir
        / "code_snapshot"
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    code_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    return (
        experiment_dir,
        result_dir,
        code_dir,
    )


# ============================================================
# 実験条件をCSV保存
# ============================================================

def save_experiment_config(
    experiment_dir,
    config,
):

    config_path = (
        Path(experiment_dir)
        / "config.csv"
    )

    with open(
        config_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "parameter",
                "value",
            ]
        )

        for key, value in config.items():

            writer.writerow(
                [
                    key,
                    value,
                ]
            )

    return config_path


# ============================================================
# 実験時点のコードをコピー
# ============================================================

def save_code_snapshot(
    source_dir,
    code_dir,
):

    source_dir = Path(
        source_dir
    )

    code_dir = Path(
        code_dir
    )

    copied_files = []

    for filename in CODE_FILES:

        source_path = (
            source_dir
            / filename
        )

        destination_path = (
            code_dir
            / filename
        )

        if source_path.exists():

            shutil.copy2(
                source_path,
                destination_path,
            )

            copied_files.append(
                destination_path
            )

    return copied_files


# ============================================================
# 実験開始時の保存処理をまとめる
# ============================================================

def initialize_experiment(
    base_dir,
    experiment_id,
    config,
):

    (
        experiment_dir,
        result_dir,
        code_dir,
    ) = create_experiment_directory(
        base_dir,
        experiment_id,
    )

    config_path = save_experiment_config(
        experiment_dir,
        config,
    )

    copied_files = save_code_snapshot(
        base_dir,
        code_dir,
    )

    print()
    print(
        "========================================"
    )
    print(
        "Experiment initialized"
    )
    print(
        "========================================"
    )

    print(
        f"Experiment ID : "
        f"{experiment_id}"
    )

    print(
        f"Experiment dir: "
        f"{experiment_dir}"
    )

    print(
        f"Result dir    : "
        f"{result_dir}"
    )

    print(
        f"Config        : "
        f"{config_path}"
    )

    print(
        "Code snapshot:"
    )

    for file_path in copied_files:
        print(
            f"  {file_path.name}"
        )

    print()

    return {
        "experiment_dir": (
            experiment_dir
        ),
        "result_dir": (
            result_dir
        ),
        "code_dir": (
            code_dir
        ),
        "config_path": (
            config_path
        ),
    }


# ============================================================
# experiment_utils.py単体で動作確認
# ============================================================

if __name__ == "__main__":

    base_dir = Path(
        __file__
    ).resolve().parent

    test_config = {
        "experiment_id": (
            "test_experiment"
        ),
        "standardization": (
            "none"
        ),
        "model": (
            "5-8-4-1"
        ),
        "optimizer": (
            "Adam"
        ),
        "learning_rate": (
            0.001
        ),
        "batch_size": (
            7
        ),
        "local_epochs": (
            10
        ),
        "server_rounds": (
            30
        ),
        "num_clients": (
            3
        ),
        "aggregation": (
            "FedAvg"
        ),
        "seed": (
            42
        ),
    }

    initialize_experiment(
        base_dir,
        "test_experiment",
        test_config,
    )
