from datetime import datetime
from pathlib import Path
import csv
import re
import shutil


# ============================================================
# 保存対象となるコード
# ============================================================

CODE_FILES = [
    "model.py",
    "task.py",
    "client_app.py",
    "server_app.py",
    "experiment_utils.py",
    "pyproject.toml",
]


# ============================================================
# 次の実験番号を自動取得
# ============================================================

def get_next_experiment_id(base_dir):
    """
    experimentsフォルダを確認し、

    exp_001
    exp_002
    exp_003

    のような既存実験番号から、
    次の番号を自動的に決める。
    """

    experiments_dir = (
        Path(base_dir)
        / "experiments"
    )

    experiments_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    experiment_numbers = []

    # --------------------------------------------------------
    # exp_001 のような名前だけを探す
    # test_experimentなどは無視する
    # --------------------------------------------------------

    pattern = re.compile(
        r"^exp_(\d{3})$"
    )

    for path in experiments_dir.iterdir():

        if not path.is_dir():
            continue

        match = pattern.match(
            path.name
        )

        if match:

            experiment_numbers.append(
                int(
                    match.group(1)
                )
            )

    # --------------------------------------------------------
    # まだ実験がなければexp_001
    # --------------------------------------------------------

    if not experiment_numbers:

        next_number = 1

    else:

        next_number = (
            max(experiment_numbers)
            + 1
        )

    experiment_id = (
        f"exp_{next_number:03d}"
    )

    return experiment_id


# ============================================================
# 実験フォルダを作成
# ============================================================

def create_experiment_directory(
    base_dir,
):

    # --------------------------------------------------------
    # 実験番号を自動決定
    # --------------------------------------------------------

    experiment_id = (
        get_next_experiment_id(
            base_dir
        )
    )

    # --------------------------------------------------------
    # 実行時刻
    # --------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    experiment_dir = (
        Path(base_dir)
        / "experiments"
        / experiment_id
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
        experiment_id,
        timestamp,
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
# 実験開始時の保存処理
# ============================================================

def initialize_experiment(
    base_dir,
    config,
):

    (
        experiment_id,
        timestamp,
        experiment_dir,
        result_dir,
        code_dir,
    ) = create_experiment_directory(
        base_dir
    )

    # --------------------------------------------------------
    # 自動生成された情報をconfigにも記録
    # --------------------------------------------------------

    config = dict(config)

    config["experiment_id"] = (
        experiment_id
    )

    config["run_timestamp"] = (
        timestamp
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
        f"Run timestamp : "
        f"{timestamp}"
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
        "experiment_id": (
            experiment_id
        ),
        "timestamp": (
            timestamp
        ),
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
# 単体動作確認
# ============================================================

if __name__ == "__main__":

    base_dir = Path(
        __file__
    ).resolve().parent

    test_config = {

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
        test_config,
    )
