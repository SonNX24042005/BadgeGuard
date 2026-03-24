from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

IMAGE_SIZE = 224

# Configuration parameters
TEST_DIR = "data/datasets/classification/cnn_format/split/test"
MODEL_PATH = "models/bg_classifier.pt"
MODEL_NAME = "resnet50"  # e.g., resnet18, resnet34, resnet50
OUTPUT_DIR = str(Path(__file__).resolve().parent / Path(__file__).stem)
BATCH_SIZE = 32
THRESHOLD = 0.6
DEVICE_ARG = "cuda"


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


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
    raise ValueError("Unsupported checkpoint format. Expected a state_dict.")


def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)

    # Average ranks for ties for a correct Mann-Whitney U based AUC.
    i = 0
    while i < sorted_scores.size:
        j = i + 1
        while j < sorted_scores.size and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[i:j] = avg_rank
        i = j

    full_ranks = np.empty_like(ranks)
    full_ranks[order] = ranks

    rank_sum_pos = float(full_ranks[pos_mask].sum())
    auc = (rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


def main() -> None:
    test_dir = Path(TEST_DIR)
    model_path = Path(MODEL_PATH)
    output_dir = Path(OUTPUT_DIR)
    batch_size = BATCH_SIZE
    threshold = THRESHOLD
    device_arg = DEVICE_ARG

    predictions_csv = output_dir / "test_predictions.csv"
    metrics_json = output_dir / "test_metrics.json"

    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = resolve_device(device_arg)
    dataset = datasets.ImageFolder(test_dir, transform=build_transform())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    if len(dataset.classes) != 2:
        raise ValueError(
            f"Expected exactly 2 classes in test set, found {len(dataset.classes)}: {dataset.classes}"
        )

    class0_name, class1_name = dataset.classes[0], dataset.classes[1]

    model = build_model(MODEL_NAME, device)
    raw_state = torch.load(model_path, map_location=device)
    state = extract_state_dict(raw_state)
    model.load_state_dict(state)
    model.eval()

    y_true_list: list[int] = []
    y_pred_list: list[int] = []
    y_score_pos_list: list[float] = []
    prediction_rows: list[str] = [
        "image_path,true_label,true_index,p_class0,p_class1,pred_label,pred_index,correct"
    ]

    sample_paths = [Path(path) for path, _ in dataset.samples]
    sample_index = 0

    with torch.inference_mode():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)

            prob_class1 = probs[:, 1]
            preds = (prob_class1 >= threshold).long()

            y_true_batch = labels.cpu().numpy().astype(np.int64)
            y_pred_batch = preds.cpu().numpy().astype(np.int64)
            y_score_batch = prob_class1.cpu().numpy().astype(np.float64)
            probs_cpu = probs.cpu().numpy().astype(np.float64)

            y_true_list.extend(y_true_batch.tolist())
            y_pred_list.extend(y_pred_batch.tolist())
            y_score_pos_list.extend(y_score_batch.tolist())

            batch_size_now = len(y_true_batch)
            for i in range(batch_size_now):
                img_path = sample_paths[sample_index + i].as_posix()
                true_idx = int(y_true_batch[i])
                pred_idx = int(y_pred_batch[i])
                p0 = float(probs_cpu[i, 0])
                p1 = float(probs_cpu[i, 1])
                true_label = dataset.classes[true_idx]
                pred_label = dataset.classes[pred_idx]
                correct = int(true_idx == pred_idx)

                prediction_rows.append(
                    f"{img_path},{true_label},{true_idx},{p0:.6f},{p1:.6f},{pred_label},{pred_idx},{correct}"
                )
            sample_index += batch_size_now

    y_true = np.array(y_true_list, dtype=np.int64)
    y_pred = np.array(y_pred_list, dtype=np.int64)
    y_score_pos = np.array(y_score_pos_list, dtype=np.float64)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    total = int(y_true.size)
    accuracy = safe_div(tp + tn, total)

    precision_pos = safe_div(tp, tp + fp)
    recall_pos = safe_div(tp, tp + fn)
    f1_pos = safe_div(2 * precision_pos * recall_pos, precision_pos + recall_pos)

    precision_neg = safe_div(tn, tn + fn)
    recall_neg = safe_div(tn, tn + fp)
    f1_neg = safe_div(2 * precision_neg * recall_neg, precision_neg + recall_neg)

    specificity = recall_neg
    balanced_accuracy = (recall_pos + recall_neg) / 2.0
    auc = binary_auc(y_true, y_score_pos)

    support_neg = int((y_true == 0).sum())
    support_pos = int((y_true == 1).sum())

    metrics = {
        "model_path": model_path.as_posix(),
        "model_name": MODEL_NAME,
        "test_dir": test_dir.as_posix(),
        "device": str(device),
        "threshold": threshold,
        "classes": {
            "class0": class0_name,
            "class1": class1_name,
        },
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "overall": {
            "num_samples": total,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "auc_roc": auc,
            "specificity": specificity,
        },
        "per_class": {
            class0_name: {
                "precision": precision_neg,
                "recall": recall_neg,
                "f1": f1_neg,
                "support": support_neg,
            },
            class1_name: {
                "precision": precision_pos,
                "recall": recall_pos,
                "f1": f1_pos,
                "support": support_pos,
            },
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv.write_text("\n".join(prediction_rows) + "\n", encoding="utf-8")
    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Device: {device}")
    print(f"Classes: {dataset.classes}")
    print(f"Samples: {total}")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(
        "Accuracy={:.4f}, BalancedAcc={:.4f}, Precision(pos)={:.4f}, Recall(pos)={:.4f}, F1(pos)={:.4f}, AUC={:.4f}".format(
            accuracy,
            balanced_accuracy,
            precision_pos,
            recall_pos,
            f1_pos,
            auc,
        )
    )
    print(f"Predictions CSV saved: {predictions_csv}")
    print(f"Metrics JSON saved: {metrics_json}")


if __name__ == "__main__":
    main()