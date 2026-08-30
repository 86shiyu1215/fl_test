from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


# ============================================================
# 基本設定
# ============================================================

RANDOM_SEED = 42

FEATURE_COLUMNS = [
    "hardness_gpa",
    "normal_load_n",
    "sliding_velocity_m_s",
    "sliding_distance_m",
    "film_thickness_nm",
]

TARGET_COLUMN = "cof"

EXPECTED_ROWS = 137

CLIENT_SIZE = 35
TEST_SIZE = 32
NUM_CLIENTS = 3


# ============================================================
# ファイルパス
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_CSV = BASE_DIR / "data" / "step3_complete_137.csv"

SPLIT_DIR = BASE_DIR / "data" / "splits"
RESULT_DIR = BASE_DIR / "results"

SPLIT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# データ読み込み
# ============================================================

df = pd.read_csv(INPUT_CSV)

print("========================================")
print("STEP3 data preparation")
print("========================================")

print(f"読み込んだデータ数: {len(df)}")


# ============================================================
# データ確認
# ============================================================

required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"必要な列がありません: {missing_columns}"
    )

if len(df) != EXPECTED_ROWS:
    raise ValueError(
        f"データ数が137行ではありません: {len(df)}行"
    )

missing_values = df[required_columns].isna().sum()

if missing_values.sum() != 0:
    raise ValueError(
        "欠損値があります:\n"
        f"{missing_values}"
    )

print("必要な6列を確認しました")
print("欠損値: 0")
print()


# ============================================================
# 元データを追跡するためのIDを追加
# ============================================================

df = df.copy()

df.insert(
    0,
    "data_id",
    range(1, len(df) + 1),
)


# ============================================================
# CoFを5つのグループに分ける
#
# 回帰なのでCoFは連続値だが、
# 小規模データでTRAIN/TESTやClient間のCoF分布が
# 極端に偏ることを防ぐため、分割時だけ5分位に分ける。
#
# このbinはDNNの特徴量としては使用しない。
# ============================================================

df["cof_bin"] = pd.qcut(
    df[TARGET_COLUMN],
    q=5,
    labels=False,
    duplicates="drop",
)


# ============================================================
# TRAIN 105件 / TEST 32件に分割
# ============================================================

train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    shuffle=True,
    stratify=df["cof_bin"],
)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"TRAIN: {len(train_df)}件")
print(f"TEST : {len(test_df)}件")


# ============================================================
# TRAIN 105件をClient 1～3へ35件ずつ分割
#
# StratifiedKFoldを利用して、
# CoFの分布が3 Clientで極端に偏らないようにする。
# ============================================================

skf = StratifiedKFold(
    n_splits=NUM_CLIENTS,
    shuffle=True,
    random_state=RANDOM_SEED,
)

client_dfs = []

for client_id, (_, client_indices) in enumerate(
    skf.split(
        train_df,
        train_df["cof_bin"],
    ),
    start=1,
):
    client_df = train_df.iloc[
        client_indices
    ].copy()

    client_df["client_id"] = client_id

    client_dfs.append(
        client_df.reset_index(drop=True)
    )


# ============================================================
# 件数確認
# ============================================================

for client_id, client_df in enumerate(
    client_dfs,
    start=1,
):
    if len(client_df) != CLIENT_SIZE:
        raise ValueError(
            f"Client {client_id} が35件ではありません: "
            f"{len(client_df)}件"
        )

    print(
        f"Client {client_id}: "
        f"{len(client_df)}件"
    )

print()


# ============================================================
# 学習用CSVを保存
#
# cof_binは分割のためだけに使ったので削除する。
# ============================================================

save_columns = (
    ["data_id"]
    + FEATURE_COLUMNS
    + [TARGET_COLUMN]
)

for client_id, client_df in enumerate(
    client_dfs,
    start=1,
):
    output_path = (
        SPLIT_DIR
        / f"client{client_id}_train.csv"
    )

    client_df[
        save_columns
    ].to_csv(
        output_path,
        index=False,
    )

test_df[
    save_columns
].to_csv(
    SPLIT_DIR / "test.csv",
    index=False,
)


# ============================================================
# 分割結果一覧を保存
# ============================================================

manifest_rows = []

for client_id, client_df in enumerate(
    client_dfs,
    start=1,
):
    temp = client_df[
        ["data_id"]
    ].copy()

    temp["split"] = "train"
    temp["client_id"] = client_id

    manifest_rows.append(temp)

test_manifest = test_df[
    ["data_id"]
].copy()

test_manifest["split"] = "test"
test_manifest["client_id"] = ""

manifest_rows.append(test_manifest)

split_manifest = pd.concat(
    manifest_rows,
    ignore_index=True,
)

split_manifest = split_manifest.sort_values(
    "data_id"
)

split_manifest.to_csv(
    RESULT_DIR / "split_manifest.csv",
    index=False,
)


# ============================================================
# 各ClientとTESTの分布を保存
# ============================================================

summary_rows = []

datasets = {
    "client1": client_dfs[0],
    "client2": client_dfs[1],
    "client3": client_dfs[2],
    "test": test_df,
}

for dataset_name, dataset_df in datasets.items():

    row = {
        "dataset": dataset_name,
        "n_samples": len(dataset_df),
    }

    for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
        row[f"{column}_mean"] = dataset_df[column].mean()
        row[f"{column}_std"] = dataset_df[column].std()
        row[f"{column}_min"] = dataset_df[column].min()
        row[f"{column}_max"] = dataset_df[column].max()

    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    RESULT_DIR / "client_distribution_summary.csv",
    index=False,
)


# ============================================================
# 完了
# ============================================================

print("データ分割が完了しました")
print()
print("保存先:")
print(SPLIT_DIR / "client1_train.csv")
print(SPLIT_DIR / "client2_train.csv")
print(SPLIT_DIR / "client3_train.csv")
print(SPLIT_DIR / "test.csv")
print(RESULT_DIR / "split_manifest.csv")
print(RESULT_DIR / "client_distribution_summary.csv")
