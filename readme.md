# Dog Pose Training Project

このリポジトリは、YOLOモデルを用いた犬のポーズ推定モデルのトレーニングプロジェクトです。

## 構成ファイル
- `dog_train.py` : モデルのトレーニングスクリプト
- `yolo11n-pose.pt` : 事前学習済みYOLOモデルの重みファイル
- `pt2onnx.py` : PyTorchの.ptファイルをONNX形式に変換するスクリプト
- `onnx_infer.py` : ONNXモデルで推論・可視化を行うスクリプト
- `runs/` : トレーニングの出力結果ディレクトリ

## 必要な環境
- Python 3.8以降
- [Ultralytics YOLO](https://docs.ultralytics.com/) ライブラリ

## セットアップ
1. 必要なライブラリのインストール
   ```bash
   pip install ultralytics
   ```
2. データセット設定ファイル（例: `dog-pose.yaml`）を用意してください。

## 使い方
1. `yolo11n-pose.pt` をプロジェクトディレクトリに配置します。
2. `dog_train.py` を実行します。
   ```bash
   python dog_train.py
   ```
3. トレーニング結果は `runs/` ディレクトリ内に保存されます。

### PyTorchモデル(.pt)からONNXへの変換
1. `pt2onnx.py` を実行します。
   ```bash
   python pt2onnx.py
   ```
2. 同じディレクトリにONNXファイルが出力されます。

`pt2onnx.py` ではultralytics YOLOのexport機能を使っています。出力ファイル名や入力サイズ等はスクリプト内で調整可能です。

### ONNXモデルで推論・可視化
1. 必要なパッケージをインストールします。
   ```bash
   pip install onnxruntime opencv-python numpy
   ```
2. 推論したい画像（例: `input.jpg`）を用意します。
3. `onnx_infer.py` を実行します。
   ```bash
   python onnx_infer.py
   ```
4. 検出結果・キーポイントが描画された画像が `output_result.jpg` として保存されます。

- 本スクリプトは出力shape (1, 300, 78)（24キーポイント: [x, y, visible]×24）に対応しています。
- キーポイント可視性が0.5以上のもののみ描画されます。
- 詳細は `onnx_infer.py` のコメント参照。
## 備考
- データセット設定ファイル（`dog-pose.yaml`）は各自で作成・配置してください。
- 詳細な使い方やカスタマイズは [Ultralytics YOLO公式ドキュメント](https://docs.ultralytics.com/) を参照してください。
