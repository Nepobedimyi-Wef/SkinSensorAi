import os
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
from tqdm import tqdm
import warnings
import json

warnings.filterwarnings('ignore')


def prepare_dataset(archive_path="archive.zip", extract_dir="dataset"):
    if not os.path.exists(archive_path):
        raise FileNotFoundError(f"Архив {archive_path} не найден!")

    if not os.path.exists(extract_dir) or len(os.listdir(extract_dir)) == 0:
        print(f"Распаковка {archive_path} в {extract_dir}...")
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Распаковка завершена.")
    else:
        print(f"Папка {extract_dir} уже существует, пропускаем распаковку.")

    items = os.listdir(extract_dir)
    top_dirs = [item for item in items if os.path.isdir(os.path.join(extract_dir, item))]

    if len(top_dirs) == 1:
        one_folder = os.path.join(extract_dir, top_dirs[0])
        sub_dirs = [d for d in os.listdir(one_folder) if os.path.isdir(os.path.join(one_folder, d))]
        if sub_dirs:
            extract_dir = one_folder
            top_dirs = sub_dirs

    print(f"Найдены папки классов: {top_dirs}")
    return extract_dir


data_path = prepare_dataset("archive.zip", "dataset")


class Config:
    DATA_DIR = data_path
    MODEL_SAVE_DIR = "models"
    MODEL_NAME = "skin_classifier.pth"
    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 25
    LEARNING_RATE = 0.001
    NUM_WORKERS = 0
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    PATIENCE = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CONFIDENCE_THRESHOLD = 0.7


config = Config()
os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)

print(f"Используемое устройство: {config.DEVICE}")
print(f"Директория с данными: {config.DATA_DIR}")

train_transforms = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_datasets():
    full_train_dataset = datasets.ImageFolder(
        root=config.DATA_DIR,
        transform=train_transforms
    )
    full_val_dataset = datasets.ImageFolder(
        root=config.DATA_DIR,
        transform=val_test_transforms
    )
    class_names = full_train_dataset.classes
    print(f"\nОбнаруженные классы: {class_names}")

    if len(class_names) < 2:
        print("\nВНИМАНИЕ: Менее двух классов. Модель не сможет различать проблемы.")
        print("Убедитесь, что в папке данных есть подпапки 'Acne' и 'normal'.")

    dataset_size = len(full_train_dataset)
    train_size = int(config.TRAIN_RATIO * dataset_size)
    val_size = int(config.VAL_RATIO * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_indices, val_indices, test_indices = random_split(
        range(dataset_size),
        [train_size, val_size, test_size]
    )

    train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_val_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_val_dataset, test_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )

    print(f"\nРазделение датасета:")
    print(f"  Всего изображений: {dataset_size}")
    print(f"  Train: {len(train_dataset)} ({train_size / dataset_size:.1%})")
    print(f"  Val:   {len(val_dataset)} ({val_size / dataset_size:.1%})")
    print(f"  Test:  {len(test_dataset)} ({test_size / dataset_size:.1%})")

    class_counts = {cls: 0 for cls in class_names}
    for idx in train_indices:
        _, label = full_train_dataset[idx]
        class_counts[class_names[label]] += 1
    print(f"\nРаспределение классов в Train:")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count} изображений")

    return train_loader, val_loader, test_loader, class_names


def create_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )
    model = model.to(config.DEVICE)
    print(f"\nМодель: ResNet-50")
    print(f"  Выходных классов: {num_classes}")
    print(f"  Параметров для обучения: {sum(p.numel() for p in model.fc.parameters())}")
    return model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        pbar.set_postfix({"loss": loss.item(), "acc": correct / total})
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pbar.set_postfix({"acc": correct / total})
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def test_model(model, loader, criterion, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    test_loss = running_loss / total
    test_acc = sum(np.array(all_preds) == np.array(all_labels)) / total
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"\n{'='*50}")
    print(f"РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Weighted F1-Score: {f1:.4f}")
    print(f"{'='*50}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(config.MODEL_SAVE_DIR, 'confusion_matrix.png'))
    plt.show()
    return test_acc, f1

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,4))
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(config.MODEL_SAVE_DIR, 'training_history.png'))
    plt.show()

def main():
    print("="*60)
    print("SkinSensorAi - Обучение классификатора акне vs норма")
    print("="*60)

    print("\n[1/5] Загрузка и подготовка данных...")
    train_loader, val_loader, test_loader, class_names = load_datasets()
    num_classes = len(class_names)

    print("\n[2/5] Создание модели...")
    model = create_model(num_classes)

    print("\n[3/5] Настройка оптимизатора...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    print("\n[4/5] Запуск обучения...")
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(config.EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{config.EPOCHS}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, config.DEVICE)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"\nРезультаты эпохи {epoch+1}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_DIR, config.MODEL_NAME))
            print(f"  ✓ Модель сохранена! (val_acc: {val_acc:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  Нет улучшения. Patience: {epochs_no_improve}/{config.PATIENCE}")

        if epochs_no_improve >= config.PATIENCE:
            print(f"\nРанняя остановка на эпохе {epoch+1}")
            break

    print("\n[5/5] Тестирование модели...")
    model.load_state_dict(torch.load(os.path.join(config.MODEL_SAVE_DIR, config.MODEL_NAME)))
    test_acc, test_f1 = test_model(model, test_loader, criterion, config.DEVICE, class_names)

    with open(os.path.join(config.MODEL_SAVE_DIR, 'classes.json'), 'w') as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    plot_training_history(history)

    print(f"\n{'='*60}")
    print(f"ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print(f"  Лучшая валидационная точность: {best_val_acc:.4f}")
    print(f"  Тестовая точность: {test_acc:.4f}")
    print(f"  Модель сохранена в: {os.path.join(config.MODEL_SAVE_DIR, config.MODEL_NAME)}")
    print(f"  Классы сохранены в: {os.path.join(config.MODEL_SAVE_DIR, 'classes.json')}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()