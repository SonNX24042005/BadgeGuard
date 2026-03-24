"""Các hàm tiện ích I/O cho ảnh và nhãn YOLO."""

from pathlib import Path

from config import IMAGE_EXTENSIONS


def get_image_files(input_dir: Path) -> list[Path]:
    """Lấy danh sách file ảnh hợp lệ trong thư mục input, sắp xếp theo tên.

    Trả về danh sách rỗng nếu thư mục không tồn tại để pipeline dừng an toàn.
    """
    if not input_dir.exists():
        return []
    images = [
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images, key=lambda p: p.name.lower())


def find_person_class_id(model_names: dict[int, str] | list[str] | tuple[str, ...]) -> int | None:
    """Tìm class id ứng với nhãn `person` trong `model.names` của Ultralytics.

    `model.names` có thể là dict hoặc list/tuple tùy model, nên cần hỗ trợ cả hai.
    """
    if isinstance(model_names, dict):
        for class_id, class_name in model_names.items():
            if str(class_name).strip().lower() == "person":
                return int(class_id)
        return None

    for class_id, class_name in enumerate(model_names):
        if str(class_name).strip().lower() == "person":
            return class_id
    return None


def write_yolo_label(label_path: Path, label_lines: list[str]) -> None:
    """Ghi file label YOLO (mỗi phần tử `label_lines` là một dòng)."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(label_lines), encoding="utf-8")
