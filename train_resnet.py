from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter
import os
import time

from utils.config_processor import config
import loaders
from resnet_k import ResNet18 as ResNet18_kuangliu

config.init(default_config_path='configs/default.yaml')

# --- Настройки ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10  # CIFAR-10 имеет 10 классов

BASE_LOG_DIR = Path(__file__).parent / config.BASE_LOG_DIR
EXPERIMENT = f'test_Resnet18_base4_kuangliu_Net' # https://github.com/kuangliu/pytorch-cifar
LOG_DIR = BASE_LOG_DIR / EXPERIMENT
assert not os.path.exists(LOG_DIR), f"Directory {LOG_DIR} already exists!"
print(str(LOG_DIR))
writer = SummaryWriter(log_dir=LOG_DIR)
    
# # --- Трансформации данных ---
# transform_train = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomCrop(32, padding=4),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
# ])

# transform_val = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
# ])

# # --- Загрузка данных ---
# train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
# val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)

# train_loader = DataLoader(train_dataset, batch_size=config.LARGE_BATCH, shuffle=True, num_workers=2)
# val_loader = DataLoader(val_dataset, batch_size=config.LARGE_BATCH, shuffle=False, num_workers=2)

train_loader, val_loader = loaders.create_cifar_loaders(config.LARGE_BATCH, use_amp=False)

# --- Создание модели ---
# model = models.resnet18()  # Используем модель без предобученных весов
# model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)  # Адаптируем для CIFAR-10
model = ResNet18_kuangliu()
model = model.to(device)

# --- Оптимизатор и функция потерь ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=config.BASE_LR, momentum=config.opt.MOMENTUM, weight_decay=config.opt.WEIGHT_DECAY)
scheduler = None # optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)  # Шаговое изменение LR
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

# --- Функция вычисления точности ---
def accuracy(output, target):
    pred = output.argmax(dim=1)
    return (pred == target).float().mean()

# --- Обучение ---
global_step = 0
for epoch in range(config.EPOCHS):
    epoch_start_time = time.time()
    model.train()
    train_loss, train_acc = 0.0, 0.0
    train_samples_no = 0
    for inputs, labels in train_loader:
        bs = len(inputs)
        train_samples_no += bs
        global_step += bs
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
        train_acc += accuracy(outputs, labels).item() * inputs.size(0)
        
    epoch_time = time.time() - epoch_start_time
    writer.add_scalar('train/epoch_time', epoch_time, global_step)
    train_loss /= train_samples_no
    train_acc /= train_samples_no

    # Валидация
    model.eval()
    val_loss, val_acc = 0.0, 0.0
    val_samples_no = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            val_samples_no += len(inputs)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            val_acc += accuracy(outputs, labels).item() * inputs.size(0)

    val_loss /= val_samples_no
    val_acc /= val_samples_no
    
    writer.add_scalar('train/loss', train_loss, global_step)
    writer.add_scalar('train/accuracy', train_acc*100, global_step)
    writer.add_scalar('test/loss', val_loss, global_step)
    writer.add_scalar('test/accuracy', val_acc*100, global_step)
    writer.add_scalar('train/epoch', epoch, global_step)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)

    print(f"Epoch {epoch+1}/{config.EPOCHS} | "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | "
          f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%")

    if scheduler is not None:
        scheduler.step()

print('done')
# # --- Сохранение модели ---
# torch.save(model.state_dict(), "resnet18_cifar10.pth")
# print("Модель сохранена как resnet18_cifar10.pth")
