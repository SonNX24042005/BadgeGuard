"""Các hàm lọc trùng cho pose và chuẩn hóa index keypoint."""

import numpy as np


def suppress_similar_pose_by_shoulders_keep_higher_shoulder_conf(
    indices: list[int],
    pose_points_xy: list[np.ndarray | None],
    pose_confs: list[np.ndarray | None],
    detection_confidences: list[float],
    shoulder_close_threshold_px: float,
) -> list[int]:
    """Lọc pose trùng nhau dựa trên khoảng cách vai trái/phải.

    Ý tưởng:
    - Nếu hai pose có vai gần nhau hơn ngưỡng, xem như cùng một người.
    - Giữ pose "tốt hơn" theo tổng confidence 2 vai; fallback sang box confidence.
    """
    if not indices:
        return []

    # Theo chuẩn YOLO Pose 17 điểm: vai trái/phải là điểm số 6 và 7 (1-based),
    # tương ứng index 5 và 6 trong mảng 0-based.
    left_shoulder_index = 5
    right_shoulder_index = 6

    def has_shoulders(points: np.ndarray | None) -> bool:
        return points is not None and len(points) > right_shoulder_index

    def shoulders_are_close(points_a: np.ndarray, points_b: np.ndarray) -> bool:
        left_dist = float(np.linalg.norm(points_a[left_shoulder_index] - points_b[left_shoulder_index]))
        right_dist = float(np.linalg.norm(points_a[right_shoulder_index] - points_b[right_shoulder_index]))
        return left_dist <= shoulder_close_threshold_px or right_dist <= shoulder_close_threshold_px

    def shoulder_score(index: int) -> float:
        confs = pose_confs[index] if index < len(pose_confs) else None
        if confs is not None and len(confs) > right_shoulder_index:
            return float(confs[left_shoulder_index] + confs[right_shoulder_index])
        return float(detection_confidences[index])

    sorted_by_quality = sorted(
        indices,
        key=lambda idx: (shoulder_score(idx), detection_confidences[idx]),
        reverse=True,
    )

    kept_indices: list[int] = []
    for candidate_index in sorted_by_quality:
        candidate_points = pose_points_xy[candidate_index] if candidate_index < len(pose_points_xy) else None
        if not has_shoulders(candidate_points):
            kept_indices.append(candidate_index)
            continue

        suppress_candidate = False
        for kept_index in kept_indices:
            kept_points = pose_points_xy[kept_index] if kept_index < len(pose_points_xy) else None
            if not has_shoulders(kept_points):
                continue
            if shoulders_are_close(candidate_points, kept_points):
                suppress_candidate = True
                break

        if not suppress_candidate:
            kept_indices.append(candidate_index)

    return sorted(kept_indices)


def normalize_pose_keypoint_indices_1_to_17(
    indices_1_to_17: list[int],
    max_keypoints: int,
) -> list[int]:
    """Chuẩn hóa danh sách keypoint từ 1-based (1..17) sang 0-based.

    Đồng thời loại phần tử ngoài phạm vi và loại trùng để đảm bảo ổn định khi vẽ.
    """
    if max_keypoints <= 0:
        return []

    normalized: list[int] = []
    seen: set[int] = set()
    for value in indices_1_to_17:
        if 1 <= int(value) <= max_keypoints:
            zero_based = int(value) - 1
            if zero_based not in seen:
                seen.add(zero_based)
                normalized.append(zero_based)

    return normalized
