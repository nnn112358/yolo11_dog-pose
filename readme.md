# Dog Pose Training Project

このリポジトリは、YOLOモデルを用いた犬のポーズ推定モデルのトレーニングプロジェクトです。

## 構成ファイル
- `dog_train.py` : モデルのトレーニングスクリプト
- `yolo11n-pose.pt` : 事前学習済みYOLOモデルの重みファイル
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

## 備考
- データセット設定ファイル（`dog-pose.yaml`）は各自で作成・配置してください。
- 詳細な使い方やカスタマイズは [Ultralytics YOLO公式ドキュメント](https://docs.ultralytics.com/) を参照してください。
