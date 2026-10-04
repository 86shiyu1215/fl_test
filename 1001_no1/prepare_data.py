from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


# ============================================================
# Settings
# ============================================================

RANDOM_SEED = 42

TEST_SIZE = 31

NUM_CLIENTS = 2


FEATURE_COLUMNS = [
    "hardness_gpa",
    "normal_load_n",
    "sliding_velocity_m_s",
    "sliding_distance_m",
    "film_thickness_nm",
]

TARGET_COLUMN = "cof"


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

SOURCE_CSV = DATA_DIR / "step3_complete_137.csv"

SPLIT_DIR = DATA_DIR / "splits"

SPLIT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Load dataset
# ============================================================

df = pd.read_csv(
    SOURCE_CSV
)


# Add data_id if it does not already exist

if "data_id" not in df.columns:

    df.insert(
        0,
        "data_id",
        np.arange(
            1,
            len(df) + 1,
        ),
    )


# ============================================================
# Check required columns
# ============================================================

required_columns = (
    FEATURE_COLUMNS
    + [TARGET_COLUMN]
)


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# Convert required columns to numeric

for column in required_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


# Check missing values

if df[required_columns].isna().any().any():

    missing = (
        df[required_columns]
        .isna()
        .sum()
    )

    raise ValueError(
        "Missing values were found:\n"
        f"{missing}"
    )


# ============================================================
# Check total samples
# ============================================================

if len(df) != 137:

    raise ValueError(
        f"Expected 137 samples, "
        f"but found {len(df)}."
    )


# ============================================================
# Train / Test split
#
# CoF distribution is stratified using 5 bins.
# ============================================================

cof_bins = pd.qcut(
    df[TARGET_COLUMN],
    q=5,
    labels=False,
    duplicates="drop",
)


all_indices = np.arange(
    len(df)
)


train_indices, test_indices = train_test_split(
    all_indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    shuffle=True,
    stratify=cof_bins,
)


train_df = (
    df.iloc[train_indices]
    .copy()
    .reset_index(drop=True)
)


test_df = (
    df.iloc[test_indices]
    .copy()
    .reset_index(drop=True)
)


# ============================================================
# Split Train 106 into 2 Clients
#
# 106 / 2 = 53 samples per client
# ============================================================

train_cof_bins = pd.qcut(
    train_df[TARGET_COLUMN],
    q=5,
    labels=False,
    duplicates="drop",
)


skf = StratifiedKFold(
    n_splits=NUM_CLIENTS,
    shuffle=True,
    random_state=RANDOM_SEED,
)


client_dfs = []


for _, client_indices in skf.split(
    train_df,
    train_cof_bins,
):

    client_df = (
        train_df.iloc[client_indices]
        .copy()
        .reset_index(drop=True)
    )

    client_dfs.append(
        client_df
    )


# ============================================================
# Check sizes
# ============================================================

if len(train_df) != 106:

    raise ValueError(
        f"Train size should be 106, "
        f"but found {len(train_df)}."
    )


if len(test_df) != 31:

    raise ValueError(
        f"Test size should be 31, "
        f"but found {len(test_df)}."
    )


for client_id, client_df in enumerate(
    client_dfs,
    start=1,
):

    if len(client_df) != 53:

        raise ValueError(
            f"Client {client_id} should have "
            f"53 samples, but found "
            f"{len(client_df)}."
        )


# ============================================================
# Save Client datasets
# ============================================================

for client_id, client_df in enumerate(
    client_dfs,
    start=1,
):

    output_path = (
        SPLIT_DIR
        / f"client{client_id}_train.csv"
    )

    client_df.to_csv(
        output_path,
        index=False,
    )


test_df.to_csv(
    SPLIT_DIR / "test.csv",
    index=False,
)


# ============================================================
# Split manifest
# ============================================================

manifest_rows = []


for client_id, client_df in enumerate(
    client_dfs,
    start=1,
):

    for data_id in client_df["data_id"]:

        manifest_rows.append(
            {
                "data_id": data_id,
                "split": f"client{client_id}",
            }
        )


for data_id in test_df["data_id"]:

    manifest_rows.append(
        {
            "data_id": data_id,
            "split": "test",
        }
    )


manifest_df = pd.DataFrame(
    manifest_rows
).sort_values(
    "data_id"
)


manifest_df.to_csv(
    SPLIT_DIR / "split_manifest.csv",
    index=False,
)


# ============================================================
# Distribution summary
# ============================================================

summary_rows = []


for client_id, client_df in enumerate(
    client_dfs,
    start=1,
):

    summary_rows.append(
        {
            "group": f"client{client_id}",
            "n": len(client_df),
            "cof_mean": client_df[TARGET_COLUMN].mean(),
            "cof_std": client_df[TARGET_COLUMN].std(),
            "cof_min": client_df[TARGET_COLUMN].min(),
            "cof_max": client_df[TARGET_COLUMN].max(),
        }
    )


summary_rows.append(
    {
        "group": "test",
        "n": len(test_df),
        "cof_mean": test_df[TARGET_COLUMN].mean(),
        "cof_std": test_df[TARGET_COLUMN].std(),
        "cof_min": test_df[TARGET_COLUMN].min(),
        "cof_max": test_df[TARGET_COLUMN].max(),
    }
)


summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(
    SPLIT_DIR
    / "client_distribution_summary.csv",
    index=False,
)


# ============================================================
# Console output
# ============================================================

print(
    "========================================"
)

print(
    "1001_no1 data split completed"
)

print(
    "========================================"
)

print(
    f"Total : {len(df)}"
)

print(
    f"Train : {len(train_df)}"
)

print(
    f"Test  : {len(test_df)}"
)


for client_id, client_df in enumerate(
    client_dfs,
    start=1,
):

    print(
        f"Client {client_id}: "
        f"{len(client_df)}"
    )


print(
    f"Seed  : {RANDOM_SEED}"
)

print(
    "========================================"
)
