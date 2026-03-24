"""Các hàm lọc box trùng/box gần nhau cho luồng detect."""

def are_boxes_close_on_requested_edges(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
    top_threshold_px: float,
    side_threshold_px: float,
) -> bool:
    """Kiểm tra hai box có "gần nhau" theo tiêu chí tùy chỉnh hay không.

    Quy tắc hiện tại:
    - Cạnh trên (y1) phải gần nhau.
    - Và ít nhất một trong hai cạnh trái/phải phải gần nhau.
    """
    x1_a, y1_a, x2_a, _ = box_a
    x1_b, y1_b, x2_b, _ = box_b

    top_close = abs(y1_a - y1_b) <= top_threshold_px
    left_close = abs(x1_a - x1_b) <= side_threshold_px
    right_close = abs(x2_a - x2_b) <= side_threshold_px

    return top_close and (left_close or right_close)


def suppress_close_boxes_keep_higher_conf(
    boxes: list[tuple[float, float, float, float]],
    confidences: list[float],
    top_threshold_px: float,
    side_threshold_px: float,
) -> list[int]:
    """Non-maximum suppression kiểu tùy biến: giữ box confidence cao hơn.

    Khác IoU-NMS truyền thống, hàm này dựa trên độ gần cạnh để loại box trùng.
    Kết quả trả về là index theo thứ tự gốc để các bước sau dùng trực tiếp.
    """
    if not boxes:
        return []

    sorted_indices = sorted(range(len(boxes)), key=lambda idx: confidences[idx], reverse=True)
    kept_indices: list[int] = []

    for candidate_index in sorted_indices:
        candidate_box = boxes[candidate_index]
        should_suppress = False

        for kept_index in kept_indices:
            if are_boxes_close_on_requested_edges(
                box_a=candidate_box,
                box_b=boxes[kept_index],
                top_threshold_px=top_threshold_px,
                side_threshold_px=side_threshold_px,
            ):
                should_suppress = True
                break

        if not should_suppress:
            kept_indices.append(candidate_index)

    return sorted(kept_indices)
