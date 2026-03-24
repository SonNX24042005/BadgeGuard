import os
import torch
import wandb
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.cuda import is_available as cuda_available
import random
from PIL import Image, ImageDraw
os.environ["WANDB_API_KEY"] = "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"

# --- Cấu hình Siêu tham số ---
DATA_DIR = 'data/datasets/classification/cnn_format/split'  # Đường dẫn tới thư mục chứa train, val, test
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
IMAGE_SIZE = 224   # Kích thước chuẩn hóa cho tất cả ảnh
MODEL_SAVE_PATH = 'models/bg_classifier.pt'
MODEL_NAME = 'resnet50'  # e.g., resnet18, resnet34, resnet50

# --- Kiểm tra Đường dẫn Trước khi Chạy ---
print("=== Đang kiểm tra hệ thống đường dẫn ===")

# 1. Kiểm tra thư mục dữ liệu
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"[Lỗi] Thư mục dữ liệu chính không tồn tại: {DATA_DIR}")

for folder in ['train', 'val', 'test']:
    folder_path = os.path.join(DATA_DIR, folder)
    if not os.path.exists(folder_path):
         raise FileNotFoundError(f"[Lỗi] Thiếu thư mục dữ liệu '{folder}' tại: {folder_path}")

# 2. Kiểm tra/Tạo thư mục lưu mô hình
save_dir = os.path.dirname(MODEL_SAVE_PATH)
if save_dir:
    if not os.path.exists(save_dir):
        print(f"-> Tạo thư mục lưu mô hình: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
    elif not os.access(save_dir, os.W_OK):
        raise PermissionError(f"[Lỗi] Không có quyền ghi vào thư mục: {save_dir}")

print("=== Kiểm tra đường dẫn hoàn tất, hệ thống sẵn sàng ===\n")

# --- Khởi tạo WandB ---
wandb.init(
    project="BadgeGuard-Classification",
    config={
        "data_dir": DATA_DIR,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "image_size": IMAGE_SIZE,
        "model_architecture": MODEL_NAME
    }
)

# --- Cấu hình Thiết bị (Device) ---
device = torch.device("cuda" if cuda_available() else "cpu")
print(f"Đang sử dụng thiết bị: {device}")

# --- Custom Transform: Random Local Rotation (RLR) ---
class RandomLocalRotation(object):
    def __init__(self, degrees=15, radius_ratio_range=(0.1, 0.3), p=0.5):
        self.degrees = degrees
        self.radius_ratio_range = radius_ratio_range
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
            
        W, H = img.size
        # Kích thước bán kính
        r_min = int(min(W, H) * self.radius_ratio_range[0])
        r_max = int(min(W, H) * self.radius_ratio_range[1])
        if r_max <= r_min:
            r_max = r_min + 5
            
        r = random.randint(r_min, r_max)
        
        # Tâm xoay (đảm bảo hình tròn nằm trong ảnh)
        cx = random.randint(r, W - r)
        cy = random.randint(r, H - r)
        
        angle = random.uniform(-self.degrees, self.degrees)
        
        # Cắt vùng
        crop_box = (cx - r, cy - r, cx + r, cy + r)
        crop = img.crop(crop_box)
        
        # Xoay vùng cắt
        rotated_crop = crop.rotate(angle, resample=Image.Resampling.BILINEAR if hasattr(Image, 'Resampling') else Image.BILINEAR)
        
        # Tạo mask hình tròn
        mask = Image.new('L', (2 * r, 2 * r), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, 2 * r, 2 * r), fill=255)
        
        # Paste trả lại ảnh cũ dùng mask
        img_copy = img.copy()
        img_copy.paste(rotated_crop, (cx - r, cy - r), mask)
        
        return img_copy

    def __repr__(self):
        return f"{self.__class__.__name__}(degrees={self.degrees}, radius_ratio_range={self.radius_ratio_range}, p={self.p})"

# --- Tiền xử lý Dữ liệu ---
# Áp dụng Resize để xử lý ảnh kích thước khác nhau
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        # RandomLocalRotation(degrees=15, radius_ratio_range=(0.1, 0.3), p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# --- Tải Dữ liệu ---
image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                  for x in ['train', 'val', 'test']}

dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
               for x in ['train', 'val']}
dataloaders['test'] = DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

print(f"Lớp phân loại: {class_names}")
print(f"Số lượng ảnh train: {dataset_sizes['train']}, val: {dataset_sizes['val']}")

# --- Kiểm soát phiên bản Dataset (Chỉ ghi chú mô tả) ---
dataset_artifact = wandb.Artifact(
    'badgeguard-dataset', 
    type='dataset', 
    description="Tập dữ liệu phân loại cho BadgeGuard đã lật ngang theo tỉ lệ 50. Đường dẫn cục bộ: " + DATA_DIR
)
dataset_artifact.metadata = {
    "num_train": dataset_sizes['train'],
    "num_val": dataset_sizes['val'],
    "classes": class_names
}
wandb.log_artifact(dataset_artifact)

# --- Xây dựng Mô hình (Sử dụng Transfer Learning) ---
print(f"-> Đang tải mô hình: {MODEL_NAME}")
if not hasattr(models, MODEL_NAME):
    raise ValueError(f"Không hỗ trợ kiến trúc mô hình: {MODEL_NAME}")

model_fn = getattr(models, MODEL_NAME)

try:
    # Thử gọi với api mới 'weights' cho torchvision >= 0.13
    model = model_fn(weights="DEFAULT")
    print("   Sử dụng weights='DEFAULT'")
except TypeError:
    # Dự phòng cho torchvision phiên bản cũ
    model = model_fn(pretrained=True)
    print("   Sử dụng pretrained=True")

# --- Cấu hình Fine-tuning ---
has_layer4 = any("layer4" in name for name, _ in model.named_parameters())

if has_layer4:
    print("-> Fine-tuning: Mở khóa các lớp cuối (layer4) của ResNet")
    for name, param in model.named_parameters():
        if "layer4" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
else:
    print("-> Fine-tuning: Mở khóa các lớp phân loại")
    for param in model.parameters():
        param.requires_grad = False

# Thay đổi lớp cuối cùng (Fully Connected) để phù hợp với bài toán nhị phân (2 lớp)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.fc.weight.requires_grad = True
model.fc.bias.requires_grad = True

model = model.to(device)

# --- Hàm Mất mát và Tối ưu hóa ---
criterion = nn.CrossEntropyLoss()

# Sử dụng AdamW với differential learning rates nếu có layer4
if has_layer4:
    optimizer = optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if "layer4" in n], 'lr': LEARNING_RATE / 10},
        {'params': model.fc.parameters(), 'lr': LEARNING_RATE}
    ], lr=LEARNING_RATE)
else:
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

# Thêm Learning Rate Scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

# Cập nhật WandB Config để theo dõi đầy đủ các biến đổi
wandb.config.update({
    "optimizer": "AdamW",
    "scheduler": "ReduceLROnPlateau",
    "has_layer4_fine_tune": has_layer4
})

# --- Quy trình Huấn luyện ---
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=10):
    best_acc = 0.0
    best_model_wts = model.state_dict()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        epoch_metrics = {"epoch": epoch}

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Chế độ huấn luyện
            else:
                model.eval()   # Chế độ đánh giá

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Ghi nhận log Metrics
            epoch_metrics[f"{phase}_loss"] = epoch_loss
            epoch_metrics[f"{phase}_acc"] = epoch_acc.item()

            if phase == 'train':
                # Ghi nhận Learning Rate để theo dõi sự suy giảm (decay)
                for i, param_group in enumerate(optimizer.param_groups):
                    epoch_metrics[f"lr_group_{i}"] = param_group['lr']

            if phase == 'val':
                # Cập nhật Scheduler
                scheduler.step(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()

        print()
        wandb.log(epoch_metrics)

    print(f'Huấn luyện hoàn tất. Độ chính xác tốt nhất: {best_acc:.4f}')
    
    # Tải trọng số tốt nhất
    model.load_state_dict(best_model_wts)
    return model, best_acc

# --- Thực thi Huấn luyện ---
trained_model, best_accuracy = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS)

# --- Đánh giá trên Tập Test ---
print("\n=== Đánh giá trên Tập Test ===")
trained_model.eval()
test_corrects = 0
test_total = len(image_datasets['test'])

with torch.no_grad():
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = trained_model(inputs)
        _, preds = torch.max(outputs, 1)
        test_corrects += torch.sum(preds == labels.data)

test_acc = test_corrects.double() / test_total
print(f"Test Accuracy: {test_acc:.4f}")
wandb.log({"test_acc": test_acc.item()})

# --- Lưu Mô hình ---
# Lưu state_dict để giảm kích thước file và linh hoạt khi tải lại
torch.save(trained_model.state_dict(), MODEL_SAVE_PATH)
print(f"Mô hình đã được lưu thành công tại: {MODEL_SAVE_PATH}")

# --- Kiểm soát phiên bản Model ---
model_artifact = wandb.Artifact('badgeguard-classifier', type='model')
model_artifact.add_file(MODEL_SAVE_PATH)
model_artifact.metadata = {"best_accuracy": best_accuracy.item() if isinstance(best_accuracy, torch.Tensor) else best_accuracy}
wandb.log_artifact(model_artifact)
print(f"Mô hình đã được tải lên WandB Artifacts.")

# Kết thúc session WandB
wandb.finish()