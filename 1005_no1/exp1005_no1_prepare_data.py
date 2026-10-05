from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


RANDOM_SEED = 42

TOTAL_SAMPLES = 92
TEST_SIZE = 20
NUM_CLIENTS = 2

TARGET_COLUMN = "target_ave_cof"


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

SOURCE_CSV = DATA_DIR / "dataset.csv"

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


if len(df) != TOTAL_SAMPLES:

    raise ValueError(
        f"Expected {TOTAL_SAMPLES} samples, "
        f"but found {len(df)}."
    )


if TARGET_COLUMN not in df.columns:

    raise ValueError(
        f"Target column '{TARGET_COLUMN}' "
        f"was not found."
    )


if df[TARGET_COLUMN].isna().any():

    raise ValueError(
        "Missing values were found "
        "in target_ave_cof."
    )


# ============================================================
# Train / Test split
#
# Target distribution is approximately preserved
# using quantile bins.
# ============================================================

target_bins = pd.qcut(
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
    stratify=target_bins,
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
# Split Train 72 into two Clients
# ============================================================

train_bins = pd.qcut(
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
    train_bins,
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
# Size checks
# ============================================================

if len(train_df) != 72:

    raise ValueError(
        f"Train size should be 72, "
        f"but found {len(train_df)}."
    )


if len(test_df) != 20:

    raise ValueError(
        f"Test size should be 20, "
        f"but found {len(test_df)}."
    )


for client_id, client_df in enumerate(
    client_dfs,
    start=1,
):

    if len(client_df) != 36:

        raise ValueError(
            f"Client {client_id} should have "
            f"36 samples, "
            f"but found {len(client_df)}."
        )


# ============================================================
# Save datasets
# ============================================================

for client_id, client_df in enumerate(
    client_dfs,
    start=1,
):

    client_df.to_csv(
        SPLIT_DIR
        / f"client{client_id}_train.csv",
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

    for _, row in client_df.iterrows():

        manifest_rows.append(
            {
                "source_row":
                    row.get(
                        "source_row",
                        np.nan,
                    ),

                "reference_group":
                    row.get(
                        "reference_group",
                        "",
                    ),

                "split":
                    f"client{client_id}",
            }
        )


for _, row in test_df.iterrows():

    manifest_rows.append(
        {
            "source_row":
                row.get(
                    "source_row",
                    np.nan,
                ),

            "reference_group":
                row.get(
                    "reference_group",
                    "",
                ),

            "split":
                "test",
        }
    )


pd.DataFrame(
    manifest_rows
).to_csv(
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
            "group":
                f"client{client_id}",

            "n":
                len(client_df),

            "cof_mean":
                client_df[
                    TARGET_COLUMN
                ].mean(),

            "cof_std":
                client_df[
                    TARGET_COLUMN
                ].std(),

            "cof_min":
                client_df[
                    TARGET_COLUMN
                ].min(),

            "cof_max":
                client_df[
                    TARGET_COLUMN
                ].max(),
        }
    )


summary_rows.append(
    {
        "group":
            "test",

        "n":
            len(test_df),

        "cof_mean":
            test_df[
                TARGET_COLUMN
            ].mean(),

        "cof_std":
            test_df[
                TARGET_COLUMN
            ].std(),

        "cof_min":
            test_df[
                TARGET_COLUMN
            ].min(),

        "cof_max":
            test_df[
                TARGET_COLUMN
            ].max(),
    }
)


pd.DataFrame(
    summary_rows
).to_csv(
    SPLIT_DIR
    / "client_distribution_summary.csv",
    index=False,
)


# ============================================================
# Console
# ============================================================

print(
    "========================================"
)

print(
    "1002_no1 data split completed"
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
