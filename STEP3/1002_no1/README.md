# 1002_no1

## 実験日
2026年10月2日

## 実験目的

共同研究先から提供された摩擦試験データについて，
NIMS-FLで使用された前処理済みデータセットをFlowerへ適用し，
DNNによる摩擦係数予測を行う．

1001_no1で使用した学習条件を基本的に維持し，
データセットを変更した場合の予測性能を確認する．


## データセット

- データセット：共同研究先データ
- 使用ファイル：dataset.csv
- 全データ数：92件
- 学習データ：72件
- テストデータ：20件
- Random seed：42
- シャッフル：あり


## Client構成

- Client数：2
- Client 1：36件
- Client 2：36件
- Test：20件


## 説明変数

合計26特徴量

### 数値特徴量
1. temperature_per_1000c
2. load_per_10n
3. log10p1_speed_per_3
4. sliding_time_per_60min

### disk材料
5. disk__40crnimoa
6. disk__alloy_800ht
7. disk__gh2132
8. disk__gh4169
9. disk__gh605
10. disk__inco718
11. disk__inco939_additive_as_built
12. disk__inco939_additive_heat_treated
13. disk__inco939_cast
14. disk__naf10
15. disk__naf15
16. disk__naf5
17. disk__rene88
18. disk__ta_w

### pin材料
19. pin__2344_h13_skd61
20. pin__alloy_800ht
21. pin__ha188
22. pin__ha25
23. pin__inco718

### 雰囲気
24. ambient__he

### 試験形式
25. test_type__pin_on_disk
26. test_type__pin_on_flat


## 目的変数

- target_ave_cof


## 学習に使用しない列

- reference_group
- source_row


## モデル

DNN

26 → 16 → 8 → 1

- Hidden activation：ReLU
- Output activation：なし


## 前処理

NIMS-FLで使用された前処理済みdataset.csvをそのまま使用する．

Flower側では追加のZ-score標準化を行わない．


## 学習条件

- Loss：MSELoss
- Optimizer：Adam
- Learning rate：0.001
- Batch size：7
- Local epoch：10
- Server round：60
- Aggregation：FedAvg
- Client参加率：100 %
- Random seed：42


## 評価指標

- MSE
- RMSE
- MAE
- R²


## 結果

- MSE：
- RMSE：
- MAE：
- R²：


## 考察

実験終了後に記入する．
