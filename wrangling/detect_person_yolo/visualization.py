"""Các hàm vẽ trực quan box/pose lên ảnh để debug nhanh chất lượng nhãn."""

import cv2
import numpy as np

from pose_filters import normalize_pose_keypoint_indices_1_to_17


def draw_boxes(
    image_bgr: np.ndarray,
    boxes: list[tuple[int, int, int, int, float, int | None]],
    show_detection_id: bool,
) -> np.ndarray:
    """Vẽ bounding box và text confidence (kèm ID nếu bật)."""
    output = image_bgr.copy()
    for x1, y1, x2, y2, confidence, detection_id in boxes:
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

        text_x = min(max(x1, 0), max(output.shape[1] - 1, 0))
        text_y = min(max(y1 - 8, 0), max(output.shape[0] - 1, 0))
        if show_detection_id and detection_id is not None:
            text = f"{detection_id} {confidence:.2f}"
        else:
            text = f"person {confidence:.2f}"
        cv2.putText(
            output,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return output


def draw_pose(
    image_bgr: np.ndarray,
    result,
    person_class_id: int | None,
    low_conf_blue_threshold: float,
    kept_result_indices: list[int] | None,
    draw_bbox: bool,
    draw_keypoints_1_to_17: list[int],
    skeleton_edges_1_to_17: list[tuple[int, int]],
) -> np.ndarray:
    """Vẽ pose theo tập keypoint được chọn và skeleton tương ứng.

    Điểm dễ nhầm:
    - `kept_result_indices` dùng để chỉ vẽ các detection đã qua bước lọc trùng.
    - Skeleton chỉ vẽ khi cả 2 đầu mút đều thuộc tập keypoint được chọn.
    """
    output = image_bgr.copy()

    if result.keypoints is None or result.keypoints.xy is None:
        return output

    keypoints_xy = result.keypoints.xy.cpu().numpy()
    keypoints_conf = None
    if hasattr(result.keypoints, "conf") and result.keypoints.conf is not None:
        keypoints_conf = result.keypoints.conf.cpu().numpy()

    class_ids = None
    xyxy_boxes = None
    box_confidences = None
    if result.boxes is not None and result.boxes.cls is not None:
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        if result.boxes.xyxy is not None:
            xyxy_boxes = result.boxes.xyxy.cpu().numpy()
        if result.boxes.conf is not None:
            box_confidences = result.boxes.conf.cpu().numpy()

    kept_result_index_set: set[int] | None = None
    if kept_result_indices is not None:
        kept_result_index_set = set(kept_result_indices)

    max_keypoints = int(keypoints_xy.shape[1]) if keypoints_xy.ndim >= 2 else 0
    selected_points = normalize_pose_keypoint_indices_1_to_17(
        indices_1_to_17=draw_keypoints_1_to_17,
        max_keypoints=max_keypoints,
    )
    selected_points_set = set(selected_points)
    selected_edges: list[tuple[int, int]] = []
    for point_a, point_b in skeleton_edges_1_to_17:
        point_a_zero_based = int(point_a) - 1
        point_b_zero_based = int(point_b) - 1
        if (
            point_a_zero_based in selected_points_set
            and point_b_zero_based in selected_points_set
            and 0 <= point_a_zero_based < max_keypoints
            and 0 <= point_b_zero_based < max_keypoints
        ):
            selected_edges.append((point_a_zero_based, point_b_zero_based))

    nose_index = 0
    left_shoulder_index = 5
    right_shoulder_index = 6

    for detection_index, person_keypoints in enumerate(keypoints_xy):
        # Nếu có danh sách đã lọc thì bỏ toàn bộ detection không được giữ lại.
        if kept_result_index_set is not None and detection_index not in kept_result_index_set:
            continue

        if person_class_id is not None and class_ids is not None:
            if detection_index >= len(class_ids) or int(class_ids[detection_index]) != person_class_id:
                continue

        selected_points_with_conf: list[tuple[int, int, float | None]] = []
        selected_all_points_xy: list[tuple[int, int]] = []
        for point_index in selected_points:
            if point_index >= len(person_keypoints):
                continue

            point_x, point_y = person_keypoints[point_index]
            point_confidence: float | None = None
            if keypoints_conf is not None and detection_index < len(keypoints_conf):
                point_confidence = float(keypoints_conf[detection_index][point_index])

            selected_all_points_xy.append((int(point_x), int(point_y)))
            selected_points_with_conf.append((int(point_x), int(point_y), point_confidence))

        if selected_all_points_xy:
            # Vẽ khung bao nhỏ (màu xanh dương) quanh các keypoint được chọn,
            # giúp quan sát nhanh vùng pose quan tâm.
            all_x = [point[0] for point in selected_all_points_xy]
            all_y = [point[1] for point in selected_all_points_xy]
            rect_x1 = min(max(min(all_x), 0), max(output.shape[1] - 1, 0))
            rect_y1 = min(max(min(all_y), 0), max(output.shape[0] - 1, 0))
            rect_x2 = min(max(max(all_x), 0), max(output.shape[1] - 1, 0))
            rect_y2 = min(max(max(all_y), 0), max(output.shape[0] - 1, 0))

            # Đồng bộ với quy tắc tạo labels_chest:
            # nếu mũi cao hơn cả 2 vai thì cắt cạnh trên xuống 2/5 khoảng cách đó.
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
                    rect_y1 = min(max(adjusted_rect_y1, 0), max(output.shape[0] - 1, 0))

            if rect_x2 > rect_x1 and rect_y2 > rect_y1:
                cv2.rectangle(output, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 0, 0), 2)

        if draw_bbox and xyxy_boxes is not None and detection_index < len(xyxy_boxes):
            x1, y1, x2, y2 = [int(value) for value in xyxy_boxes[detection_index]]
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if box_confidences is not None and detection_index < len(box_confidences):
                text = f"person {float(box_confidences[detection_index]):.2f}"
                text_x = min(max(x1, 0), max(output.shape[1] - 1, 0))
                text_y = min(max(y1 - 8, 0), max(output.shape[0] - 1, 0))
                cv2.putText(
                    output,
                    text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        for joint_a, joint_b in selected_edges:
            if joint_a >= len(person_keypoints) or joint_b >= len(person_keypoints):
                continue

            x_a, y_a = person_keypoints[joint_a]
            x_b, y_b = person_keypoints[joint_b]
            cv2.line(output, (int(x_a), int(y_a)), (int(x_b), int(y_b)), (0, 255, 255), 2)

        for point_x, point_y, point_confidence in selected_points_with_conf:
            # Quy ước màu:
            # - Đỏ: keypoint có confidence đủ tốt.
            # - Tím: confidence thấp hơn ngưỡng cảnh báo.
            point_color = (0, 0, 255)
            if point_confidence is not None and point_confidence < low_conf_blue_threshold:
                point_color = (255, 0, 255)
            cv2.circle(output, (point_x, point_y), 3, point_color, -1)

    return output
