from __future__ import annotations

"""Wrapper tương thích ngược cho cách chạy cũ `python detect_person_yolo.py`.

Script này không chứa logic detect/pose nữa, chỉ chuyển điều khiển sang
project module mới trong thư mục `detect_person_yolo/`.
"""

from pathlib import Path
import runpy
import sys


def main() -> None:
	"""Nạp đường dẫn project module và chạy file `cli.py` như chương trình chính.

	Điểm khó hiểu:
	- `runpy.run_path` không tự thêm thư mục chứa `cli.py` vào `sys.path`.
	- Vì vậy cần chèn `project_dir` trước để các import kiểu `from pipeline import ...`
	  trong nội bộ project hoạt động đúng.
	"""
	project_dir = Path(__file__).resolve().parent / "detect_person_yolo"
	project_entry = project_dir / "cli.py"
	project_dir_str = str(project_dir)
	if project_dir_str not in sys.path:
		sys.path.insert(0, project_dir_str)
	runpy.run_path(str(project_entry), run_name="__main__")


if __name__ == "__main__":
	main()