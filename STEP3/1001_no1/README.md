# 1001_no1

## 実験日
2026年10月1日

## 実験目的

文献から収集した137件のDLC摩擦試験データを用いて，
Client数を2拠点とした連合学習を実施する．

今後，3 Client条件との比較を行い，
Client数および1 Clientあたりのデータ数の違いが
Global modelのCoF予測性能に与える影響を検討する．


## データセット

- データセット：文献由来DLC摩擦試験データ
- 全データ数：137件
- 学習データ：106件
- テストデータ：31件
- Random seed：42
- データ分割時にシャッフル：あり


## Client構成

- Client数：2
- Client 1：53件
- Client 2：53件
- Test：31件


## 説明変数

1. hardness_gpa
2. normal_load_n
3. sliding_velocity_m_s
4. sliding_distance_m
5. film_thickness_nm


## 目的変数

- cof


## モデル

- モデル：MLP / DNN
- 入力層：5
- 第1隠れ層：16
- 第2隠れ層：8
- 出力層：1

構造：

5 → 16 → 8 → 1

- 活性化関数：ReLU
- 出力層活性化関数：なし


## 前処理

- Z-score標準化：あり
- 標準化対象：5つの説明変数
- 標準化パラメータ：Train 106件のみから算出
- 全Clientで共通の平均値・標準偏差を使用
- TestにはTrainから求めた平均値・標準偏差を適用
- 目的変数cof：標準化なし


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


## 今回変更した主な条件

以前の3 Client条件からClient数を変更する．

- Client数：3 → 2
- 1 Clientあたりの学習データ数：53件
- Train：106件
- Test：31件


## 結果

- MSE：
- RMSE：
- MAE：
- R²：


## 考察

実験終了後に記入する．
