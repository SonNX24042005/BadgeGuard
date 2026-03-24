"""Cấu hình trung tâm cho pipeline detect/pose.

Mục tiêu file này:
- Gom toàn bộ hằng số để chỉnh ở một nơi.
- Tránh rải giá trị magic number trong nhiều module.
- Giữ nguyên hành vi cũ của script gốc.
"""

from pathlib import Path

# =========================
# Cấu hình
# =========================
INPUT_DIR = Path("data/extracted_frames/images")
LABELS_DIR = Path("data/extracted_frames/labels")
LABELS_POSE_DIR = Path("data/extracted_frames/labels_pose")
LABELS_CHEST_DIR = Path("data/extracted_frames/labels_chest")
VISUALIZE_DIR = Path("data/extracted_frames/vis_every_10")

# Đường dẫn model (.pt) cho cả biến thể detect và pose (ví dụ: yolo11x.pt, yolo11x-pose.pt)
MODEL_PATH = "models/yolo11x.pt"

# ---------- Tham số cho model Detect ----------
# Ngưỡng độ tin cậy khi suy luận với model detect thường
DETECT_CONF_THRESHOLD = 0.25
# Kích thước ảnh đầu vào khi suy luận với model detect thường
DETECT_IMGSZ = 1920

# ---------- Tham số cho model Pose ---------
# Ngưỡng độ tin cậy khi suy luận pose (giảm ngưỡng giúp giữ các pose khó/bị che khuất)
POSE_CONF_THRESHOLD = 0.25
# Kích thước ảnh đầu vào khi suy luận pose (tăng kích thước giúp bắt keypoint nhỏ tốt hơn)
POSE_IMGSZ = 1920

# Lưu 1 ảnh trực quan hóa sau mỗi N ảnh được xử lý
SAVE_VIS_EVERY_N = 10
# Có thêm ID tăng dần vào dòng nhãn (chỉ dùng cho nhãn detect)
WITH_ID = False
# Ngưỡng gần nhau theo cạnh trên (pixel) để lọc box trùng trong luồng detect
TOP_EDGE_CLOSE_THRESHOLD_PX = 12.0
# Ngưỡng gần nhau theo cạnh trái/phải (pixel) để lọc box trùng trong luồng detect
SIDE_EDGE_CLOSE_THRESHOLD_PX = 12.0
# Nếu confidence keypoint < ngưỡng này thì chấm keypoint sẽ vẽ màu tím (ngược lại màu đỏ)
POSE_KPT_LOW_CONF_BLUE_THRESHOLD = 0.50
# Bật/tắt vẽ bounding box khi dùng model pose
POSE_DRAW_BBOX = True
# Chọn keypoint cần vẽ theo thứ tự 1..17 (theo chuẩn YOLO pose 17 keypoints)
# Ví dụ: [1, 2, 3, 4, 5] hoặc [6, 7, 12, 13]
POSE_DRAW_KEYPOINTS_1_TO_17 = [1, 6, 7, 12, 13]
# Skeleton theo thứ tự 1..17; chỉ cạnh có đủ 2 đầu mút nằm trong POSE_DRAW_KEYPOINTS_1_TO_17 mới được vẽ
POSE_SKELETON_EDGES_1_TO_17 = [
    (16, 14),
    (14, 12),
    (17, 15),
    (15, 13),
    (12, 13),
    (6, 12),
    (7, 13),
    (6, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (9, 11),
    (2, 3),
    (1, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (5, 7),
]
# Ngưỡng gần nhau của vai theo pixel để lọc pose trùng
POSE_SHOULDER_CLOSE_THRESHOLD_PX = 40.0

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Lưu ý:
# - Danh sách keypoint đang dùng chuẩn YOLO Pose 17 điểm theo thứ tự 1..17.
# - Khi xử lý nội bộ trong code sẽ chuyển về chỉ số 0-based (0..16).
