from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# =========================
# Configuration
# =========================
# Thư mục chứa video đầu vào cần quét.
INPUT_DIR = Path("data/raw_video")
# Thư mục lưu các frame được trích xuất.
OUTPUT_DIR = Path("data/extracted_frames")

# Danh sách đuôi video được chấp nhận khi duyệt thư mục đầu vào.
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v"}

# Lấy mẫu theo chu kỳ: chỉ xử lý mỗi N frame (ví dụ 10 nghĩa là frame 0, 10, 20, ...).
SAMPLE_EVERY_N = 12
# Ngưỡng chuyển động (% pixel thay đổi) để quyết định có lưu frame gốc hay không.
MOTION_THRESHOLD_PERCENT = 5.0

# Kích thước resize dùng riêng cho bước so sánh chuyển động (không ảnh hưởng chất lượng ảnh lưu).
RESIZE_WIDTH = 320
RESIZE_HEIGHT = 180
# Kích thước kernel blur Gaussian (nên là số lẻ, code sẽ tự hiệu chỉnh nếu nhập sai).
BLUR_KERNEL_SIZE = 5
# Ngưỡng cường độ cho ảnh sai khác (absdiff) trước khi đếm pixel thay đổi.
DIFF_INTENSITY_THRESHOLD = 30

# Định dạng ảnh đầu ra (thường dùng .jpg hoặc .png).
IMAGE_EXT = ".jpg"
# Chất lượng JPEG (1-100), chỉ áp dụng khi IMAGE_EXT là .jpg/.jpeg.
JPEG_QUALITY = 100


def ensure_valid_blur_kernel(kernel_size: int) -> int:
	if kernel_size < 1:
		return 1
	if kernel_size % 2 == 0:
		return kernel_size + 1
	return kernel_size


def get_video_files(input_dir: Path) -> list[Path]:
	if not input_dir.exists():
		return []
	files = [
		path
		for path in input_dir.iterdir()
		if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
	]
	return sorted(files, key=lambda p: p.name.lower())


def preprocess_for_motion(frame_bgr: np.ndarray, blur_kernel_size: int) -> np.ndarray:
	resized = cv2.resize(frame_bgr, (RESIZE_WIDTH, RESIZE_HEIGHT), interpolation=cv2.INTER_AREA)
	gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)
	return blurred


def compute_motion_ratio_percent(
	current_processed: np.ndarray,
	reference_processed: np.ndarray,
	diff_intensity_threshold: int,
) -> float:
	diff = cv2.absdiff(current_processed, reference_processed)
	_, binary = cv2.threshold(diff, diff_intensity_threshold, 255, cv2.THRESH_BINARY)
	changed_pixels = int(np.count_nonzero(binary))
	total_pixels = binary.size
	if total_pixels == 0:
		return 0.0
	return (changed_pixels / total_pixels) * 100.0


def save_original_frame(frame_bgr: np.ndarray, output_path: Path) -> bool:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	if IMAGE_EXT.lower() in {".jpg", ".jpeg"}:
		return bool(cv2.imwrite(str(output_path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]))
	return bool(cv2.imwrite(str(output_path), frame_bgr))


def process_single_video(video_path: Path, output_dir: Path, blur_kernel_size: int) -> tuple[bool, int]:
	saved_count = 0
	decode_failed_count = 0
	cap = None

	try:
		cap = cv2.VideoCapture(str(video_path))
		if not cap.isOpened():
			raise RuntimeError("Cannot open video file")

		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		if total_frames <= 0:
			total_frames = None

		reference_processed = None
		frame_index = 0

		with tqdm(
			total=total_frames,
			desc=f"Frames: {video_path.name}",
			unit="frame",
			leave=False,
		) as frame_bar:
			frame_bar.set_postfix(saved=saved_count)
			while True:
				if total_frames is not None and frame_index >= total_frames:
					break

				grabbed = cap.grab()
				if not grabbed:
					break

				if frame_index % SAMPLE_EVERY_N == 0:
					ok, frame_bgr = cap.retrieve()
					if not ok:
						decode_failed_count += 1
					else:
						current_processed = preprocess_for_motion(frame_bgr, blur_kernel_size)
						output_name = f"{video_path.stem}_frame_{frame_index:08d}{IMAGE_EXT}"
						output_path = output_dir / output_name

						if reference_processed is None:
							if save_original_frame(frame_bgr, output_path):
								saved_count += 1
								frame_bar.set_postfix(saved=saved_count)
								reference_processed = current_processed
						else:
							motion_ratio = compute_motion_ratio_percent(
								current_processed=current_processed,
								reference_processed=reference_processed,
								diff_intensity_threshold=DIFF_INTENSITY_THRESHOLD,
							)
							if motion_ratio > MOTION_THRESHOLD_PERCENT:
								if save_original_frame(frame_bgr, output_path):
									saved_count += 1
									frame_bar.set_postfix(saved=saved_count)
									reference_processed = current_processed

				frame_index += 1
				frame_bar.update(1)

		if decode_failed_count > 0:
			tqdm.write(
				f"[WARN] {video_path.name}: bo qua {decode_failed_count} frame loi decode, tiep tuc xu ly."
			)

	except Exception as error:
		print(f"[ERROR] Failed to process '{video_path.name}': {error}")
		return False, 0
	finally:
		if cap is not None:
			cap.release()

	return True, saved_count


def main() -> None:
	blur_kernel_size = ensure_valid_blur_kernel(BLUR_KERNEL_SIZE)
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	video_files = get_video_files(INPUT_DIR)
	if not video_files:
		print(f"No video files found in: {INPUT_DIR.resolve()}")
		return

	total_saved_frames = 0
	success_videos = 0
	failed_videos = 0

	for video_path in tqdm(video_files, desc="Videos", unit="video"):
		success, saved_now = process_single_video(
			video_path=video_path,
			output_dir=OUTPUT_DIR,
			blur_kernel_size=blur_kernel_size,
		)
		total_saved_frames += saved_now
		tqdm.write(f"{video_path.name}: da luu {saved_now} frame")

		if success:
			success_videos += 1
		else:
			failed_videos += 1

	print("\n=== Extraction Summary ===")
	print(f"Input directory  : {INPUT_DIR.resolve()}")
	print(f"Output directory : {OUTPUT_DIR.resolve()}")
	print(f"Processed videos : {len(video_files)}")
	print(f"Saved frames     : {total_saved_frames}")
	print(f"Successful videos: {success_videos}")
	print(f"Failed videos    : {failed_videos}")


if __name__ == "__main__":
	main()
