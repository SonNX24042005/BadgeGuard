"""Xây dựng dòng nhãn YOLO và dữ liệu box/pose phục vụ visualize."""

from pathlib import Path

from bbox_filters import suppress_close_boxes_keep_higher_conf
from config import (
    POSE_SHOULDER_CLOSE_THRESHOLD_PX,
    POSE_DRAW_KEYPOINTS_1_TO_17,
    SIDE_EDGE_CLOSE_THRESHOLD_PX,
    TOP_EDGE_CLOSE_THRESHOLD_PX,
)
from pose_filters import normalize_pose_keypoint_indices_1_to_17, suppress_similar_pose_by_shoulders_keep_higher_shoulder_conf


def detect_model_output_type(model, model_path: str) -> str:
    """Xác định model đang chạy theo nhánh `detect` hay `pose`.

    Ưu tiên `model.task`; fallback theo tên file model để tăng độ an toàn.
    """
    model_task = str(getattr(model, "task", "")).strip().lower()
    if model_task == "pose":
        return "pose"

    model_stem = Path(model_path).stem.lower()
    if "pose" in model_stem:
        return "pose"

    return "detect"


def build_bbox_lines_and_boxes(
    result,
    image_width: int,
    image_height: int,
    person_class_id: int | None,
    model_output_type: str,
    include_detection_id: bool,
    start_detection_id: int,
) -> tuple[list[str], list[tuple[int, int, int, int, float, int | None]], int, list[int]]:
    """Tạo label YOLO + box vẽ + danh sách index detection đã giữ lại.

    Luồng chính:
    1) Lọc chỉ class `person`.
    2) (Nếu pose) thu thập keypoint để ghi label và lọc trùng.
    3) Lọc box trùng theo cạnh.
    4) (Nếu pose) lọc tiếp theo độ gần vai.
    5) Chuẩn hóa bbox sang định dạng YOLO và sinh dòng nhãn.
    """
    label_lines: list[str] = []
    visual_boxes: list[tuple[int, int, int, int, float, int | None]] = []
    next_detection_id = start_detection_id
    kept_result_indices: list[int] = []

    if result.boxes is None:
        return label_lines, visual_boxes, next_detection_id, kept_result_indices

    boxes = result.boxes
    if boxes.xyxy is None or boxes.cls is None or boxes.conf is None:
        return label_lines, visual_boxes, next_detection_id, kept_result_indices

    xyxy_boxes = boxes.xyxy.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy().astype(int)
    confidences = boxes.conf.cpu().numpy()

    pose_xyn = None
    pose_xy = None
    pose_conf = None
    if model_output_type == "pose" and result.keypoints is not None:
        if result.keypoints.xy is not None:
            pose_xy = result.keypoints.xy.cpu().numpy()
        if result.keypoints.xyn is not None:
            pose_xyn = result.keypoints.xyn.cpu().numpy()
        if hasattr(result.keypoints, "conf") and result.keypoints.conf is not None:
            pose_conf = result.keypoints.conf.cpu().numpy()

    candidate_boxes: list[tuple[float, float, float, float]] = []
    candidate_confidences: list[float] = []
    candidate_pose_lines: list[list[float] | None] = []
    # Hai danh sách dưới dùng cho bước lọc pose trùng theo vai.
    candidate_pose_points_xy: list = []
    candidate_pose_confs: list = []
    # Mapping quan trọng: candidate_index -> detection_index gốc của model output.
    # Nhờ đó sau khi lọc, ta vẫn biết detection nào cần vẽ ở bước visualize.
    candidate_result_indices: list[int] = []

    for detection_index, (xyxy, class_id, confidence) in enumerate(zip(xyxy_boxes, class_ids, confidences)):
        if person_class_id is None:
            continue
        if int(class_id) != person_class_id:
            continue

        x1, y1, x2, y2 = [float(value) for value in xyxy]
        x1 = min(max(x1, 0.0), float(image_width))
        y1 = min(max(y1, 0.0), float(image_height))
        x2 = min(max(x2, 0.0), float(image_width))
        y2 = min(max(y2, 0.0), float(image_height))

        candidate_boxes.append((x1, y1, x2, y2))
        candidate_confidences.append(float(confidence))
        candidate_result_indices.append(detection_index)

        pose_line: list[float] | None = None
        pose_points_for_filter_xy = None
        pose_confs_for_filter = None
        if model_output_type == "pose":
            if pose_xy is not None and detection_index < len(pose_xy):
                pose_points_for_filter_xy = pose_xy[detection_index]
            if pose_xyn is not None and detection_index < len(pose_xyn):
                pose_points = pose_xyn[detection_index]
                # `pose_xyn` đã normalized [0..1], đúng format YOLO pose label.
                pose_visibility = pose_conf[detection_index] if pose_conf is not None and detection_index < len(pose_conf) else None
                pose_confs_for_filter = pose_visibility

                pose_line = []
                for point_index, (kp_x, kp_y) in enumerate(pose_points):
                    kp_x = min(max(float(kp_x), 0.0), 1.0)
                    kp_y = min(max(float(kp_y), 0.0), 1.0)
                    pose_line.extend([kp_x, kp_y])

                    if pose_visibility is not None:
                        visibility_conf = min(max(float(pose_visibility[point_index]), 0.0), 1.0)
                        pose_line.append(visibility_conf)

        candidate_pose_lines.append(pose_line)
        candidate_pose_points_xy.append(pose_points_for_filter_xy)
        candidate_pose_confs.append(pose_confs_for_filter)

    kept_candidate_indices = suppress_close_boxes_keep_higher_conf(
        boxes=candidate_boxes,
        confidences=candidate_confidences,
        top_threshold_px=TOP_EDGE_CLOSE_THRESHOLD_PX,
        side_threshold_px=SIDE_EDGE_CLOSE_THRESHOLD_PX,
    )

    if model_output_type == "pose":
        # Lọc 2 lớp: sau khi lọc box gần nhau, lọc tiếp bằng vai để giảm duplicate pose.
        kept_candidate_indices = suppress_similar_pose_by_shoulders_keep_higher_shoulder_conf(
            indices=kept_candidate_indices,
            pose_points_xy=candidate_pose_points_xy,
            pose_confs=candidate_pose_confs,
            detection_confidences=candidate_confidences,
            shoulder_close_threshold_px=POSE_SHOULDER_CLOSE_THRESHOLD_PX,
        )

    for candidate_index in kept_candidate_indices:
        if candidate_index < len(candidate_result_indices):
            kept_result_indices.append(candidate_result_indices[candidate_index])

        x1, y1, x2, y2 = candidate_boxes[candidate_index]
        confidence = candidate_confidences[candidate_index]

        bbox_width = max(x2 - x1, 0.0)
        bbox_height = max(y2 - y1, 0.0)
        x_center = x1 + (bbox_width / 2.0)
        y_center = y1 + (bbox_height / 2.0)

        x_center_norm = x_center / max(image_width, 1)
        y_center_norm = y_center / max(image_height, 1)
        width_norm = bbox_width / max(image_width, 1)
        height_norm = bbox_height / max(image_height, 1)

        detection_id: int | None = None
        if include_detection_id:
            detection_id = next_detection_id
            next_detection_id += 1

        if include_detection_id and detection_id is not None:
            # Format mở rộng cho detect: thêm ID tracking nội bộ ở cuối dòng.
            label_lines.append(
                f"{person_class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f} {detection_id}"
            )
        else:
            if model_output_type == "pose":
                # Pose label: class + bbox + toàn bộ keypoint (x, y, [conf]).
                pose_values = candidate_pose_lines[candidate_index] if candidate_index < len(candidate_pose_lines) else None
                if pose_values:
                    pose_text = " ".join(f"{value:.6f}" for value in pose_values)
                    label_lines.append(
                        f"{person_class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f} {pose_text}"
                    )
                else:
                    label_lines.append(
                        f"{person_class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
                    )
            else:
                label_lines.append(
                    f"{person_class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
                )

        visual_boxes.append((int(x1), int(y1), int(x2), int(y2), float(confidence), detection_id))

    return label_lines, visual_boxes, next_detection_id, kept_result_indices


def build_chest_bbox_label_lines(
    result,
    image_width: int,
    image_height: int,
    person_class_id: int | None,
    kept_result_indices: list[int] | None,
) -> list[str]:
    """Tạo nhãn bbox ngực từ hình chữ nhật bao các keypoint được chọn.

    Format trả về giống nhãn detect YOLO:
    class_id x_center y_center width height (đều normalized [0..1]).
    """
    label_lines: list[str] = []

    if person_class_id is None:
        return label_lines

    if result.keypoints is None or result.keypoints.xy is None:
        return label_lines

    keypoints_xy = result.keypoints.xy.cpu().numpy()

    class_ids = None
    if result.boxes is not None and result.boxes.cls is not None:
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

    kept_result_index_set: set[int] | None = None
    if kept_result_indices is not None:
        kept_result_index_set = set(kept_result_indices)

    max_keypoints = int(keypoints_xy.shape[1]) if keypoints_xy.ndim >= 2 else 0
    selected_points = normalize_pose_keypoint_indices_1_to_17(
        indices_1_to_17=POSE_DRAW_KEYPOINTS_1_TO_17,
        max_keypoints=max_keypoints,
    )

    nose_index = 0
    left_shoulder_index = 5
    right_shoulder_index = 6

    for detection_index, person_keypoints in enumerate(keypoints_xy):
        if kept_result_index_set is not None and detection_index not in kept_result_index_set:
            continue

        if class_ids is not None:
            if detection_index >= len(class_ids) or int(class_ids[detection_index]) != person_class_id:
                continue

        selected_points_xy: list[tuple[int, int]] = []
        for point_index in selected_points:
            if point_index >= len(person_keypoints):
                continue
            point_x, point_y = person_keypoints[point_index]
            selected_points_xy.append((int(point_x), int(point_y)))

        if not selected_points_xy:
            continue

        all_x = [point[0] for point in selected_points_xy]
        all_y = [point[1] for point in selected_points_xy]
        rect_x1 = min(max(min(all_x), 0), max(image_width - 1, 0))
        rect_y1 = min(max(min(all_y), 0), max(image_height - 1, 0))
        rect_x2 = min(max(max(all_x), 0), max(image_width - 1, 0))
        rect_y2 = min(max(max(all_y), 0), max(image_height - 1, 0))

        # Quy tắc rút ngắn cạnh đứng phía trên:
        # Nếu mũi cao hơn cả 2 vai, tăng y_top thêm 2/5 khoảng cách từ mũi tới vai cao nhất.
        # Nếu thiếu mũi/vai thì giữ nguyên bbox như cũ.
        if (
            len(person_keypoints) > right_shoulder_index
            and len(person_keypoints) > nose_index
        ):
            nose_y = float(person_keypoints[nose_index][1])
            left_shoulder_y = float(person_keypoints[left_shoulder_index][1])
            right_shoulder_y = float(person_keypoints[right_shoulder_index][1])
            highest_shoulder_y = min(left_shoulder_y, right_shoulder_y)

            if nose_y < highest_shoulder_y:
                delta = (2.0 / 5.0) * (highest_shoulder_y - nose_y)
                adjusted_rect_y1 = int(round(rect_y1 + delta))
                rect_y1 = min(max(adjusted_rect_y1, 0), max(image_height - 1, 0))

        if rect_x2 <= rect_x1 or rect_y2 <= rect_y1:
            continue

        bbox_width = float(rect_x2 - rect_x1)
        bbox_height = float(rect_y2 - rect_y1)
        x_center = float(rect_x1) + (bbox_width / 2.0)
        y_center = float(rect_y1) + (bbox_height / 2.0)

        x_center_norm = x_center / max(image_width, 1)
        y_center_norm = y_center / max(image_height, 1)
        width_norm = bbox_width / max(image_width, 1)
        height_norm = bbox_height / max(image_height, 1)

        label_lines.append(
            f"{person_class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
        )

    return label_lines
