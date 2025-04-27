import onnxruntime as ort
import numpy as np
import cv2

# 入力画像パスとONNXファイル名を指定
ONNX_PATH = "yolo11n-pose.onnx"  # 変換後のONNXファイル名に合わせて変更してください
IMAGE_PATH = "input.jpg"          # 推論したい画像ファイル

# 画像の前処理（YOLO用640x640, 正規化など）
def preprocess(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img_np = img.astype(np.float32) / 255.0  # 0-1正規化
    img_np = np.transpose(img_np, (2, 0, 1))  # CHW
    img_np = np.expand_dims(img_np, axis=0)   # バッチ次元
    return img_np

# ONNXモデルロード
def inference_onnx(onnx_path, input_tensor):
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    return outputs

if __name__ == "__main__":
    img_tensor = preprocess(IMAGE_PATH)
    preds = inference_onnx(ONNX_PATH, img_tensor)
    print("ONNX推論結果 shape:", np.array(preds[0]).shape)

    # 元画像（OpenCV BGR）を読み込み
    orig_img = cv2.imread(IMAGE_PATH)
    orig_h, orig_w = orig_img.shape[:2]
    scale_x = orig_w / 640
    scale_y = orig_h / 640

    # 推論出力 shape: (1, 300, 78)
    #   - 1: バッチサイズ（通常は1）
    #   - 300: 最大検出数（NMS後の検出上限）
    #   - 78: 各検出の属性ベクトル
    #         [x1, y1, x2, y2, score, class_id,
    #          kpt1_x, kpt1_y, kpt1_visible,
    #          kpt2_x, kpt2_y, kpt2_visible,
    #          ...（全24キーポイント分）]
    #   - x1, y1, x2, y2: バウンディングボックス座標（入力画像サイズ基準, 0~640）
    #   - score: 検出信頼度
    #   - class_id: 検出クラスID
    #   - kptN_x, kptN_y, kptN_visible: N番目キーポイント座標(x, y)と可視性(0~1)
    dets = preds[0][0]  # (300, 78)
    for i, det in enumerate(dets):
        score = det[4]
        cls = int(det[5])
        if score < 0.3:
            continue  # スコア閾値でフィルタ
        bbox = det[:4]
        keypoints = det[6:]
        print(f"Detection {i}: score={score:.2f}, class={cls}, bbox={bbox}")
        # bboxを元画像スケールに戻して描画
        x1, y1, x2, y2 = bbox
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)
        cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # キーポイントがある場合、表示（24個なら 24x3）
        if len(keypoints) == 24 * 3:
            kpt_arr = keypoints.reshape(24, 3)
            for j, (x, y, visible) in enumerate(kpt_arr):
                px = int(x * scale_x)
                py = int(y * scale_y)
                if visible > 0.5:
                    cv2.circle(orig_img, (px, py), 3, (0, 0, 255), -1)
                print(f"  kpt{j}: ({x:.1f}, {y:.1f}), visible={visible:.2f}")

    # 結果画像を保存
    out_path = "output_result.jpg"
    cv2.imwrite(out_path, orig_img)
    print(f"可視化画像を {out_path} に保存しました。")
