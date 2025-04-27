from ultralytics import YOLO

# モデルのロード
yolo_model = YOLO("yolo11n-pose.pt")

# ONNXエクスポート
# 出力ファイル名や入力サイズは必要に応じて調整してください
yolo_model.export(format="onnx", imgsz=640, nms=True, opset=17, simplify=True)

print("ONNXエクスポートが完了しました。")
