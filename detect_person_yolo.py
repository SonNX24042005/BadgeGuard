from __future__ import annotations

from pathlib import Path
import shutil

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


# =========================
# Cấu hình
# =========================
INPUT_DIR = Path("data/extracted_frames/images")
LABELS_DIR = Path("data/extracted_frames/labels")
LABELS_POSE_DIR = Path("data/extracted_frames/labels_pose")
VISUALIZE_DIR = Path("data/extracted_frames/vis_every_10")

# Đường dẫn model (.pt) cho cả biến thể detect và pose (ví dụ: yolo11x.pt, yolo11x-pose.pt)
MODEL_PATH = "yolo11x-pose.pt"

# ---------- Tham số cho model Detect ----------
# Ngưỡng độ tin cậy khi suy luận với model detect thường
DETECT_CONF_THRESHOLD = 0.25
# Kích thước ảnh đầu vào khi suy luận với model detect thường
DETECT_IMGSZ = 640

# ---------- Tham số cho model Pose ----------
# Ngưỡng độ tin cậy khi suy luận pose (giảm ngưỡng giúp giữ các pose khó/bị che khuất)
POSE_CONF_THRESHOLD = 0.10
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
# Ngưỡng confidence tối thiểu để vẽ keypoint và skeleton pose
POSE_KPT_CONF_THRESHOLD = 0.15
# Bật/tắt vẽ bounding box khi dùng model pose
POSE_DRAW_BBOX = True
# Chọn keypoint cần vẽ theo thứ tự 1..17 (theo chuẩn YOLO pose 17 keypoints)
# Ví dụ: [1, 2, 3, 4, 5] hoặc [6, 7, 12, 13]
POSE_DRAW_KEYPOINTS_1_TO_17 = [ 6, 7, 8, 9, 10, 11, 12, 13]
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


def get_image_files(input_dir: Path) -> list[Path]:
	if not input_dir.exists():
		return []
	images = [
		path
		for path in input_dir.iterdir()
		if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
	]
	return sorted(images, key=lambda p: p.name.lower())


def find_person_class_id(model_names: dict[int, str] | list[str] | tuple[str, ...]) -> int | None:
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
	label_path.parent.mkdir(parents=True, exist_ok=True)
	label_path.write_text("\n".join(label_lines), encoding="utf-8")


def are_boxes_close_on_requested_edges(
	box_a: tuple[float, float, float, float],
	box_b: tuple[float, float, float, float],
	top_threshold_px: float,
	side_threshold_px: float,
) -> bool:
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


def suppress_similar_pose_by_shoulders_keep_higher_shoulder_conf(
	indices: list[int],
	pose_points_xy: list[np.ndarray | None],
	pose_confs: list[np.ndarray | None],
	detection_confidences: list[float],
	shoulder_close_threshold_px: float,
) -> list[int]:
	if not indices:
		return []

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


def detect_model_output_type(model, model_path: str) -> str:
	model_task = str(getattr(model, "task", "")).strip().lower()
	if model_task == "pose":
		return "pose"

	model_stem = Path(model_path).stem.lower()
	if "pose" in model_stem:
		return "pose"

	return "detect"


def normalize_pose_keypoint_indices_1_to_17(
	indices_1_to_17: list[int],
	max_keypoints: int,
) -> list[int]:
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


def build_bbox_lines_and_boxes(
	result,
	image_width: int,
	image_height: int,
	person_class_id: int | None,
	model_output_type: str,
	include_detection_id: bool,
	start_detection_id: int,
) -> tuple[list[str], list[tuple[int, int, int, int, float, int | None]], int, list[int]]:
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
	candidate_pose_points_xy: list[np.ndarray | None] = []
	candidate_pose_confs: list[np.ndarray | None] = []
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
		pose_points_for_filter_xy: np.ndarray | None = None
		pose_confs_for_filter: np.ndarray | None = None
		if model_output_type == "pose":
			if pose_xy is not None and detection_index < len(pose_xy):
				pose_points_for_filter_xy = pose_xy[detection_index]
			if pose_xyn is not None and detection_index < len(pose_xyn):
				pose_points = pose_xyn[detection_index]
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
			label_lines.append(
				f"{person_class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f} {detection_id}"
			)
		else:
			if model_output_type == "pose":
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


def draw_boxes(
	image_bgr: np.ndarray,
	boxes: list[tuple[int, int, int, int, float, int | None]],
	show_detection_id: bool,
) -> np.ndarray:
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
	kpt_conf_threshold: float,
	kept_result_indices: list[int] | None,
	draw_bbox: bool,
	draw_keypoints_1_to_17: list[int],
	skeleton_edges_1_to_17: list[tuple[int, int]],
) -> np.ndarray:
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

	for detection_index, person_keypoints in enumerate(keypoints_xy):
		if kept_result_index_set is not None and detection_index not in kept_result_index_set:
			continue

		if person_class_id is not None and class_ids is not None:
			if detection_index >= len(class_ids) or int(class_ids[detection_index]) != person_class_id:
				continue

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

			if keypoints_conf is not None and detection_index < len(keypoints_conf):
				if (
					float(keypoints_conf[detection_index][joint_a]) < kpt_conf_threshold
					or float(keypoints_conf[detection_index][joint_b]) < kpt_conf_threshold
				):
					continue

			x_a, y_a = person_keypoints[joint_a]
			x_b, y_b = person_keypoints[joint_b]
			cv2.line(output, (int(x_a), int(y_a)), (int(x_b), int(y_b)), (0, 255, 255), 2)

		for point_index in selected_points:
			if point_index >= len(person_keypoints):
				continue

			point_x, point_y = person_keypoints[point_index]
			if keypoints_conf is not None and detection_index < len(keypoints_conf):
				if float(keypoints_conf[detection_index][point_index]) < kpt_conf_threshold:
					continue

			cv2.circle(output, (int(point_x), int(point_y)), 3, (0, 0, 255), -1)

	return output


def main() -> None:
	include_detection_id = WITH_ID

	labels_output_dir = LABELS_DIR
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
		labels_output_dir = LABELS_POSE_DIR
		include_detection_id = False
		inference_conf_threshold = POSE_CONF_THRESHOLD
		inference_imgsz = POSE_IMGSZ
	else:
		inference_conf_threshold = DETECT_CONF_THRESHOLD
		inference_imgsz = DETECT_IMGSZ

	if labels_output_dir.exists():
		shutil.rmtree(labels_output_dir)

	labels_output_dir.mkdir(parents=True, exist_ok=True)
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

		image_bgr = cv2.imread(str(image_path))
		if image_bgr is None:
			decode_failed += 1
			write_yolo_label(label_path, [])
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

		if index % SAVE_VIS_EVERY_N == 0:
			if model_output_type == "pose":
				visualized = draw_pose(
					image_bgr=image_bgr,
					result=result,
					person_class_id=person_class_id,
					kpt_conf_threshold=POSE_KPT_CONF_THRESHOLD,
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
	print(f"Visualize directory   : {VISUALIZE_DIR.resolve()}")
	print(f"Processed images      : {total_images}")
	print(f"Decode failed images  : {decode_failed}")
	print(f"Images with person    : {images_with_person}")
	print(f"Total person boxes    : {total_person_boxes}")
	print(f"Detection ID enabled  : {include_detection_id}")
	print(f"Visualization saved   : {visual_saved} (every {SAVE_VIS_EVERY_N} images)")


if __name__ == "__main__":
	main()