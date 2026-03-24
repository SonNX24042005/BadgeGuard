# detect_person_yolo

Tài liệu ngắn mô tả kiến trúc sau khi tách từ script monolithic.

## 1) Mục tiêu

Project này chạy pipeline YOLO để:
- đọc ảnh từ thư mục input,
- suy luận người (detect hoặc pose),
- ghi nhãn YOLO theo từng ảnh,
- (với pose) ghi thêm nhãn `labels_chest` từ hình chữ nhật bao các keypoint đã chọn,
- lưu ảnh visualize định kỳ.

## 2) Cách chạy

### Cách tương thích cũ
```bash
python3 detect_person_yolo.py
```

### Cách chạy trực tiếp module mới
```bash
python3 detect_person_yolo/cli.py
```

## 3) Luồng thực thi tổng quát

1. `detect_person_yolo.py` (wrapper) thêm đường dẫn project vào `sys.path`.
2. Wrapper gọi `detect_person_yolo/cli.py` bằng `runpy`.
3. `cli.py` gọi `pipeline.main()`.
4. `pipeline.main()`:
   - nạp model,
   - xác định nhánh `detect` hay `pose`,
   - lặp qua từng ảnh,
   - gọi `build_bbox_lines_and_boxes()` để tạo nhãn,
   - gọi `draw_boxes()` hoặc `draw_pose()` để visualize định kỳ,
   - in summary cuối cùng.

## 4) Vai trò từng file

- `config.py`
  - Toàn bộ cấu hình đường dẫn và ngưỡng.
  - Chỉnh tham số chạy ở đây.
  - Có thêm `LABELS_CHEST_DIR` cho nhãn bbox ngực.

- `io_utils.py`
  - `get_image_files()`: lấy danh sách ảnh hợp lệ.
  - `find_person_class_id()`: tìm class id của `person` trong model names.
  - `write_yolo_label()`: ghi file nhãn YOLO.

- `bbox_filters.py`
  - Lọc box trùng theo tiêu chí gần cạnh (`top/left/right`) và giữ box có confidence cao hơn.

- `pose_filters.py`
  - Lọc pose trùng theo khoảng cách vai.
  - Chuẩn hóa danh sách keypoint từ 1-based sang 0-based.

- `labeling.py`
  - `detect_model_output_type()`: xác định nhánh model.
  - `build_bbox_lines_and_boxes()`: lõi tạo dòng label YOLO + dữ liệu box phục vụ visualize.
  - `build_chest_bbox_label_lines()`: tạo nhãn bbox từ hình chữ nhật bao keypoint (format như detect YOLO).

- `visualization.py`
  - `draw_boxes()`: vẽ bbox và text confidence/id.
  - `draw_pose()`: vẽ keypoint/skeleton/bbox cho pose.

- `pipeline.py`
  - Điều phối toàn bộ quy trình chạy.

- `cli.py`
  - Entry script của project module mới.

## 5) Các điểm dễ nhầm

- Mapping index:
  - `candidate_index` là index sau khi lọc ứng viên.
  - `detection_index` là index gốc từ output model.
  - Cần map ngược để chỉ vẽ đúng detection đã giữ.

- Chuẩn tọa độ:
  - bbox trong label dùng giá trị normalized `[0..1]`.
  - pose dùng `xyn` (normalized), không dùng `xy` (pixel) khi ghi label.

- Lọc trùng 2 lớp:
  1. Lọc box gần nhau theo cạnh.
  2. Với pose, lọc thêm theo độ gần vai.

- Làm mới dữ liệu khi chạy pose:
  - Mỗi lần chạy pose sẽ xóa cả `labels_pose` và `labels_chest` trước khi lưu mới.

- Đọc ảnh lỗi:
  - Vẫn ghi file label rỗng để giữ tương ứng 1 ảnh ↔ 1 file nhãn.

- Visualize định kỳ:
  - Chỉ lưu mỗi `SAVE_VIS_EVERY_N` ảnh để giảm I/O và dung lượng.

## 6) Chỉnh cấu hình thường dùng

Sửa trong `config.py`:
- `MODEL_PATH`
- `DETECT_CONF_THRESHOLD`, `DETECT_IMGSZ`
- `POSE_CONF_THRESHOLD`, `POSE_IMGSZ`
- `SAVE_VIS_EVERY_N`
- `POSE_DRAW_KEYPOINTS_1_TO_17`
- `POSE_SKELETON_EDGES_1_TO_17`

## 7) Dependency

Dùng dependency từ file gốc `requirements.txt` của workspace.

Nếu chưa cài:
```bash
pip install -r requirements.txt
```
