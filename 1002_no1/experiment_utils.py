import csv
import re
import shutil

from datetime import datetime
from pathlib import Path


CODE_FILES = [
    "model.py",
    "task.py",
    "client_app.py",
    "server_app.py",
    "experiment_utils.py",
    "prepare_data.py",
    "pyproject.toml",
    "README.md",
]


DATA_FILES = [
    "dataset.csv",
]


SPLIT_FILES = [
    "client1_train.csv",
    "client2_train.csv",
    "test.csv",
    "split_manifest.csv",
    "client_distribution_summary.csv",
]


def get_next_experiment_id(
    base_dir: Path,
):

    experiments_dir = (
        base_dir
        / "experiments"
    )


    experiments_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    pattern = re.compile(
        r"^exp_(\d{3})$"
    )


    existing_numbers = []


    for path in experiments_dir.iterdir():

        if not path.is_dir():

            continue


        match = pattern.match(
            path.name
        )


        if match:

            existing_numbers.append(
                int(
                    match.group(1)
                )
            )


    next_number = (
        max(existing_numbers) + 1
        if existing_numbers
        else 1
    )


    return (
        f"exp_{next_number:03d}"
    )


def create_experiment(
    base_dir: Path,
    config: dict,
):

    experiment_id = (
        get_next_experiment_id(
            base_dir
        )
    )


    experiment_dir = (
        base_dir
        / "experiments"
        / experiment_id
    )


    code_snapshot_dir = (
        experiment_dir
        / "code_snapshot"
    )


    data_snapshot_dir = (
        experiment_dir
        / "data_snapshot"
    )


    results_dir = (
        experiment_dir
        / "results"
    )


    code_snapshot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    data_snapshot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    for filename in CODE_FILES:

        source = (
            base_dir
            / filename
        )


        if source.exists():

            shutil.copy2(
                source,
                code_snapshot_dir
                / filename,
            )


    for filename in DATA_FILES:

        source = (
            base_dir
            / "data"
            / filename
        )


        if source.exists():

            shutil.copy2(
                source,
                data_snapshot_dir
                / filename,
            )


    split_snapshot_dir = (
        data_snapshot_dir
        / "splits"
    )


    split_snapshot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    for filename in SPLIT_FILES:

        source = (
            base_dir
            / "data"
            / "splits"
            / filename
        )


        if source.exists():

            shutil.copy2(
                source,
                split_snapshot_dir
                / filename,
            )


    config_to_save = dict(
        config
    )


    config_to_save[
        "experiment_id"
    ] = experiment_id


    config_to_save[
        "run_timestamp"
    ] = datetime.now().isoformat(
        timespec="seconds"
    )


    with open(
        experiment_dir
        / "config.csv",
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.writer(
            file
        )


        writer.writerow(
            [
                "parameter",
                "value",
            ]
        )


        for key, value in config_to_save.items():

            writer.writerow(
                [
                    key,
                    value,
                ]
            )


    return (
        experiment_id,
        experiment_dir,
        results_dir,
    )
