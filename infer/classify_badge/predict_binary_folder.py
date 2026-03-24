from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


import os
import wandb

# 1. THIẾT LẬP API KEY
# (Bạn có thể lấy key của mình tại: https://wandb.ai/authorize)
os.environ["WANDB_API_KEY"] = ""

# 2. Khởi tạo W&B API (Nó sẽ tự động dùng key từ biến môi trường ở trên)
api = wandb.Api()

# ĐỊNH NGHĨA THÔNG TIN DỰ ÁN
entity = "sonnx24042005-hanoi-university-of-science-and-technology" 
project = "BadgeGuard-Classification"
model_name = "badgeguard-classifier"
MODEL_VERSION = "v1"  # "v1", "v2", "latest", etc.

# --- TẢI MÔ HÌNH TỪ WANDB ---
path_v1 = f"{entity}/{project}/{model_name}:{MODEL_VERSION}"
artifact_v1 = api.artifact(path_v1)
download_dir = artifact_v1.download(root=f'./models/badgeguard_{MODEL_VERSION}')
print(f"Đã tải xong {MODEL_VERSION}!")





IMAGE_SIZE = 224
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# --- Configuration Parameters ---
# Parameters have been moved here from the main function.
# Do not use CLI tool to parse parameters, just edit directly here.
INPUT_DIR = Path(r"data/datasets/classification/test_real")
MODEL_NAME = "resnet50"  # Kiến trúc mặc định
try:
	run_v1 = artifact_v1.logged_by()
	if run_v1 and "model_architecture" in run_v1.config:
		MODEL_NAME = run_v1.config["model_architecture"]
		print(f"-> Tự động nhận diện kiến trúc mô hình từ WandB: {MODEL_NAME}")
except Exception as e:
	print(f"-> Không thể lấy kiến trúc mô hình từ WandB, sử dụng mặc định: {MODEL_NAME}")

# Tìm file mô hình trong thư mục tải về
downloaded_files = list(Path(download_dir).glob("*.pt"))
if downloaded_files:
	MODEL_PATH = downloaded_files[0]
	print(f"-> Sử dụng mô hình tải về: {MODEL_PATH}")
else:
	MODEL_PATH = Path(r"models/bg_classifier.pt")
	print(f"-> Không tìm thấy mô hình tải về, dùng mặc định: {MODEL_PATH}")

_script_path = Path(__file__).resolve()
_output_dir = _script_path.parent / _script_path.stem
OUTPUT_CSV = _output_dir / "predictions_binary.csv"
SUMMARY_JSON = _output_dir / "predictions_binary_summary.json"
BATCH_SIZE = 64
NUM_WORKERS = 4
THRESHOLD = 0.5
CLASS0_NAME = "badge"
CLASS1_NAME = "chest"
DEVICE_ARG = "cuda"  # auto | cpu | cuda
RECURSIVE = True


def resolve_device(device_arg: str) -> torch.device:
	if device_arg == "cpu":
		return torch.device("cpu")
	if device_arg == "cuda":
		if not torch.cuda.is_available():
			raise RuntimeError("CUDA requested but not available")
		return torch.device("cuda")
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transform() -> transforms.Compose:
	return transforms.Compose(
		[
			transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
		]
	)


def build_model(model_name: str, device: torch.device) -> nn.Module:
	if not hasattr(models, model_name):
		raise ValueError(f"Unsupported model architecture: {model_name}")
	model_fn = getattr(models, model_name)
	model = model_fn(weights=None)
	num_ftrs = model.fc.in_features
	model.fc = nn.Linear(num_ftrs, 2)
	return model.to(device)


def extract_state_dict(raw_state: object) -> dict[str, torch.Tensor]:
	if isinstance(raw_state, dict):
		if "state_dict" in raw_state and isinstance(raw_state["state_dict"], dict):
			return raw_state["state_dict"]
		if "model_state_dict" in raw_state and isinstance(raw_state["model_state_dict"], dict):
			return raw_state["model_state_dict"]
		if all(isinstance(v, torch.Tensor) for v in raw_state.values()):
			return raw_state  # type: ignore[return-value]
	raise ValueError("Unsupported checkpoint format. Expected a state_dict-like checkpoint.")


def collect_image_paths(input_dir: Path, recursive: bool) -> list[Path]:
	iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
	image_paths = [p for p in iterator if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
	image_paths.sort()
	return image_paths


class FolderImageDataset(Dataset[tuple[torch.Tensor, str]]):
	def __init__(self, image_paths: list[Path], transform: transforms.Compose) -> None:
		self.image_paths = image_paths
		self.transform = transform

	def __len__(self) -> int:
		return len(self.image_paths)

	def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
		path = self.image_paths[index]
		try:
			with Image.open(path) as image:
				rgb = image.convert("RGB")
		except (UnidentifiedImageError, OSError) as exc:
			raise RuntimeError(f"Cannot read image: {path}") from exc
		return self.transform(rgb), path.as_posix()


def main() -> None:
	if not INPUT_DIR.exists():
		raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")
	if not MODEL_PATH.exists():
		raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
	if BATCH_SIZE <= 0:
		raise ValueError("batch_size must be > 0")
	if NUM_WORKERS < 0:
		raise ValueError("num_workers must be >= 0")
	if not 0.0 <= THRESHOLD <= 1.0:
		raise ValueError("threshold must be in [0, 1]")

	device = resolve_device(DEVICE_ARG)

	image_paths = collect_image_paths(INPUT_DIR, recursive=RECURSIVE)
	if not image_paths:
		raise ValueError(f"No supported images found in: {INPUT_DIR}")

	dataset = FolderImageDataset(image_paths=image_paths, transform=build_transform())
	dataloader = DataLoader(
		dataset,
		batch_size=BATCH_SIZE,
		shuffle=False,
		num_workers=NUM_WORKERS,
		pin_memory=(device.type == "cuda"),
	)

	model = build_model(MODEL_NAME, device)
	raw_state = torch.load(MODEL_PATH, map_location=device)
	state_dict = extract_state_dict(raw_state)
	model.load_state_dict(state_dict)
	model.eval()

	OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
	SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)

	counts = {CLASS0_NAME: 0, CLASS1_NAME: 0}
	rows: list[dict[str, object]] = []

	with torch.inference_mode():
		for inputs, paths in dataloader:
			inputs = inputs.to(device)

			logits = model(inputs)
			probs = torch.softmax(logits, dim=1)

			prob_class1 = probs[:, 1]
			preds = (prob_class1 >= THRESHOLD).long()

			probs_cpu = probs.cpu()
			preds_cpu = preds.cpu().tolist()

			for idx, image_path in enumerate(paths):
				pred_idx = int(preds_cpu[idx])
				p0 = float(probs_cpu[idx, 0].item())
				p1 = float(probs_cpu[idx, 1].item())
				pred_label = CLASS0_NAME if pred_idx == 0 else CLASS1_NAME
				counts[pred_label] += 1

				rows.append(
					{
						"image_path": image_path,
						"pred_index": pred_idx,
						"pred_label": pred_label,
						"p_class0": p0,
						"p_class1": p1,
						"threshold": THRESHOLD,
					}
				)

	with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(
			f,
			fieldnames=[
				"image_path",
				"pred_index",
				"pred_label",
				"p_class0",
				"p_class1",
				"threshold",
			],
		)
		writer.writeheader()
		writer.writerows(rows)

	summary = {
		"model_path": MODEL_PATH.as_posix(),
		"model_name": MODEL_NAME,
		"input_dir": INPUT_DIR.as_posix(),
		"device": str(device),
		"threshold": THRESHOLD,
		"classes": {
			"class0": CLASS0_NAME,
			"class1": CLASS1_NAME,
		},
		"num_images": len(rows),
		"pred_counts": counts,
		"output_csv": OUTPUT_CSV.as_posix(),
	}

	SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

	print(f"Device: {device}")
	print(f"Images: {len(rows)}")
	print(f"Pred counts: {counts}")
	print(f"CSV saved: {OUTPUT_CSV}")
	print(f"Summary saved: {SUMMARY_JSON}")


if __name__ == "__main__":
	main()
