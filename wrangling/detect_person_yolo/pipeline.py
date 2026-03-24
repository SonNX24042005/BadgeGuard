"""Pipeline chính: đọc ảnh -> suy luận YOLO -> ghi nhãn -> lưu ảnh visualize."""

import shutil

import cv2
from tqdm import tqdm
from ultralytics import YOLO

from config import (
    DETECT_CONF_THRESHOLD,
    DETECT_IMGSZ,
    INPUT_DIR,
    LABELS_CHEST_DIR,
    LABELS_DIR,
    LABELS_POSE_DIR,
    MODEL_PATH,
    POSE_CONF_THRESHOLD,
    POSE_DRAW_BBOX,
    POSE_DRAW_KEYPOINTS_1_TO_17,
    POSE_KPT_LOW_CONF_BLUE_THRESHOLD,
    POSE_SKELETON_EDGES_1_TO_17,
    POSE_IMGSZ,
    SAVE_VIS_EVERY_N,
    VISUALIZE_DIR,
    WITH_ID,
)
from io_utils import find_person_class_id, get_image_files, write_yolo_label
from labeling import build_bbox_lines_and_boxes, build_chest_bbox_label_lines, detect_model_output_type
from visualization import draw_boxes, draw_pose


def main() -> None:
    """Chạy toàn bộ quy trình detect/pose trên thư mục ảnh đầu vào.

    Hàm này là entrypoint logic thực tế của project sau khi tách module.
    """
    include_detection_id = WITH_ID

    labels_output_dir = LABELS_DIR
    chest_labels_output_dir = LABELS_CHEST_DIR
    # Dọn thư mục visualize cũ để tránh lẫn kết quả giữa các lần chạy.
    if VISUALIZE_DIR.exists():
        shutil.rmtree(VISUALIZE_DIR)

    image_files = get_image_files(INPUT_DIR)
    if not image_files:
        print(f"No image files found in: {INPUT_DIR.resolve()}")
        return

    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    model_output_type = detect_model_output_type(model=model, model_path=MODEL_PATH)

    if model_output_type == "pose":
        # Luồng pose luôn tắt detection ID để giữ đúng format pose label YOLO.
        labels_output_dir = LABELS_POSE_DIR
        include_detection_id = False
        inference_conf_threshold = POSE_CONF_THRESHOLD
        inference_imgsz = POSE_IMGSZ
    else:
        inference_conf_threshold = DETECT_CONF_THRESHOLD
        inference_imgsz = DETECT_IMGSZ

    if model_output_type == "pose":
        # Yêu cầu làm mới dữ liệu mỗi lần chạy pose:
        # xóa cả labels_pose và labels_chest trước khi ghi lại.
        if LABELS_POSE_DIR.exists():
            shutil.rmtree(LABELS_POSE_DIR)
        if chest_labels_output_dir.exists():
            shutil.rmtree(chest_labels_output_dir)
    else:
        if labels_output_dir.exists():
            shutil.rmtree(labels_output_dir)

    labels_output_dir.mkdir(parents=True, exist_ok=True)
    if model_output_type == "pose":
        chest_labels_output_dir.mkdir(parents=True, exist_ok=True)
    VISUALIZE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Model output type: {model_output_type}")
    print(f"Inference conf/imgsz: {inference_conf_threshold} / {inference_imgsz}")

    person_class_id = find_person_class_id(model.names)
    if person_class_id is None:
        print("[WARN] Class 'person' not found in model names. Label files will be empty.")
    else:
        print(f"Class 'person' detected with class_id={person_class_id}")

    total_images = 0
    decode_failed = 0
    images_with_person = 0
    total_person_boxes = 0
    visual_saved = 0
    next_detection_id = 1

    for index, image_path in enumerate(tqdm(image_files, desc=f"Detect {model_output_type}", unit="image"), start=1):
        total_images += 1
        label_path = labels_output_dir / f"{image_path.stem}.txt"
        chest_label_path = chest_labels_output_dir / f"{image_path.stem}.txt"

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            # Nếu không đọc được ảnh vẫn tạo file nhãn rỗng để giữ 1-1 ảnh/label.
            decode_failed += 1
            write_yolo_label(label_path, [])
            if model_output_type == "pose":
                write_yolo_label(chest_label_path, [])
            continue

        result = model.predict(
            source=str(image_path),
            conf=inference_conf_threshold,
            imgsz=inference_imgsz,
            verbose=False,
        )[0]

        image_height, image_width = image_bgr.shape[:2]
        label_lines, visual_boxes, next_detection_id, kept_result_indices = build_bbox_lines_and_boxes(
            result=result,
            image_width=image_width,
            image_height=image_height,
            person_class_id=person_class_id,
            model_output_type=model_output_type,
            include_detection_id=include_detection_id,
            start_detection_id=next_detection_id,
        )

        if label_lines:
            images_with_person += 1
            total_person_boxes += len(label_lines)

        write_yolo_label(label_path, label_lines)

        if model_output_type == "pose":
            chest_label_lines = build_chest_bbox_label_lines(
                result=result,
                image_width=image_width,
                image_height=image_height,
                person_class_id=person_class_id,
                kept_result_indices=kept_result_indices,
            )
            write_yolo_label(chest_label_path, chest_label_lines)

        if index % SAVE_VIS_EVERY_N == 0:
            # Chỉ lưu visualize theo chu kỳ để giảm I/O và dung lượng.
            if model_output_type == "pose":
                visualized = draw_pose(
                    image_bgr=image_bgr,
                    result=result,
                    person_class_id=person_class_id,
                    low_conf_blue_threshold=POSE_KPT_LOW_CONF_BLUE_THRESHOLD,
                    kept_result_indices=kept_result_indices,
                    draw_bbox=POSE_DRAW_BBOX,
                    draw_keypoints_1_to_17=POSE_DRAW_KEYPOINTS_1_TO_17,
                    skeleton_edges_1_to_17=POSE_SKELETON_EDGES_1_TO_17,
                )
            else:
                visualized = draw_boxes(
                    image_bgr=image_bgr,
                    boxes=visual_boxes,
                    show_detection_id=include_detection_id,
                )
            visual_output = VISUALIZE_DIR / image_path.name
            cv2.imwrite(str(visual_output), visualized)
            visual_saved += 1

    print(f"\n=== YOLO11 {model_output_type} Summary ===")
    print(f"Input directory       : {INPUT_DIR.resolve()}")
    print(f"Labels directory      : {labels_output_dir.resolve()}")
    if model_output_type == "pose":
        print(f"Chest labels directory: {chest_labels_output_dir.resolve()}")
    print(f"Visualize directory   : {VISUALIZE_DIR.resolve()}")
    print(f"Processed images      : {total_images}")
    print(f"Decode failed images  : {decode_failed}")
    print(f"Images with person    : {images_with_person}")
    print(f"Total person boxes    : {total_person_boxes}")
    print(f"Detection ID enabled  : {include_detection_id}")
    print(f"Visualization saved   : {visual_saved} (every {SAVE_VIS_EVERY_N} images)")
