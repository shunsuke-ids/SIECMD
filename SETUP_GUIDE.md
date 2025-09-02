# SIECMD セットアップ・実行ガイド

## 概要
SIECMD（Single Image Estimation of Cell Migration Direction）は、単一画像から細胞の移動方向を深層円形回帰で推定するプロジェクトです。

## セットアップ手順

### 1. 依存関係のインストール
```bash
# 仮想環境を有効化
source venv/bin/activate

# 必要なパッケージをインストール
pip install numpy scipy opencv-python tensorflow keras keras-cv
```

### 2. 環境変数の設定
```bash
# プロジェクトルートでPythonパスを設定
export PYTHONPATH=$(pwd)
```

### 3. データセットの準備
データセットは以下の形式で準備してください：
```
data_dir/
├── 0/           # 0度方向の画像
│   ├── image1.png
│   └── image2.png
├── 1/           # 1度方向の画像
│   ├── image3.png
│   ...
...
├── 359/         # 359度方向の画像
│   ├── ...
```

### 4. 利用可能なスクリプト

#### A. 環境セットアップ
```bash
chmod +x setup_env.sh
source setup_env.sh
```

#### B. データセット前処理
```bash
./prepare_dataset.sh <元データディレクトリ> <データセット名> <保存ディレクトリ>
```

例：
```bash
./prepare_dataset.sh ./raw_data my_dataset ./prepared_data
```

#### C. プロービングモデルの訓練
```bash
./run_probing.sh <前処理済みデータディレクトリ> <データセット名>
```

例：
```bash
./run_probing.sh prepared_data my_dataset
```

#### D. ファインチューニングモデルの訓練
```bash
./run_fine_tuning.sh <前処理済みデータディレクトリ> <データセット名>
```

例：
```bash
./run_fine_tuning.sh prepared_data my_dataset
```

## 基本的な使用例

### サンプルデータでのテスト
```bash
# 1. サンプルデータを生成
python3 create_sample_data.py

# 2. データセットを前処理
./prepare_dataset.sh sample_data test_dataset prepared_data

# 3. プロービングモデルを訓練
source venv/bin/activate
export PYTHONPATH=$(pwd)
python3 src/regression/probing.py $(pwd) prepared_data test_dataset --n 2 --epochs 5
```

### 手動実行
```bash
# 仮想環境を有効化
source venv/bin/activate

# Pythonパスを設定
export PYTHONPATH=$(pwd)

# プロービングモデルを訓練
python3 src/regression/probing.py $(pwd) <data_dir> <dataset_name> [オプション]

# ファインチューニングモデルを訓練
python3 src/regression/fine_tuning.py $(pwd) <data_dir> <dataset_name> [オプション]
```

## オプション

### probing.py のオプション
- `--n`, `-n`: クロスバリデーションの回数（デフォルト: 4）
- `--epochs`, `-e`: エポック数（デフォルト: 50）
- `--batch_size`, `-b`: バッチサイズ（デフォルト: 32）
- `--prediction_tolerance`, `-pt`: 予測許容範囲（デフォルト: 45）

### fine_tuning.py のオプション
- `--n`, `-n`: クロスバリデーションの回数（デフォルト: 4）
- `--epochs`, `-e`: 初期訓練エポック数（デフォルト: 10）
- `--fine_tuning_epochs`, `-fte`: ファインチューニングエポック数（デフォルト: 50）
- `--batch_size`, `-b`: バッチサイズ（デフォルト: 32）
- `--prediction_tolerance`, `-pt`: 予測許容範囲（デフォルト: 45）

## トラブルシューティング

### CUDA警告について
GPU が利用できない環境では CUDA 関連の警告が出ますが、CPU での実行は正常に行われます。

### Keras バージョンの問題
このプロジェクトは Keras 3.x に対応するよう修正済みです。もし問題が発生した場合は：
```bash
pip install tensorflow keras keras-cv --upgrade
```

## 出力

- 訓練済みモデルは `weights/` ディレクトリに保存されます
- プロービングモデル: `weights/weights_*.weights.h5`
- ファインチューニングモデル: `weights/fine_tuning/<backbone_name>/weights<dataset>_*.weights.h5`

## 論文情報
詳細は以下の論文を参照してください：
[Single Image Estimation of Cell Migration Direction by Deep Circular Regression](https://arxiv.org/abs/2406.19162) (L. Bruns et al.)
