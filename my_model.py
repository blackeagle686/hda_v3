import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0
from torch.utils.data import DataLoader, random_split
from PIL import Image

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HDAImgTrainer:
    # Transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Init
    def __init__(
        self,
        train_path: str,
        val_path: str | None = None,
        val_split: float = 0.2,
        batch_size: int = 32,
        checkpoint_dir: str = "checkpoints"
    ):
        #  Dataset base (no transform yet)
        base_dataset = datasets.ImageFolder(train_path)
        self.class_names = base_dataset.classes
        num_classes = len(self.class_names)

        print(f"[INFO] Detected classes: {self.class_names}")

        #  Train / Val split
        if val_path is None:
            val_size = int(len(base_dataset) * val_split)
            train_size = len(base_dataset) - val_size
            train_idx, val_idx = random_split(
                range(len(base_dataset)),
                [train_size, val_size]
            )

            train_dataset = datasets.ImageFolder(
                train_path, transform=self.train_transforms
            )
            val_dataset = datasets.ImageFolder(
                train_path, transform=self.val_transforms
            )

            train_dataset.samples = [base_dataset.samples[i] for i in train_idx.indices]
            val_dataset.samples = [base_dataset.samples[i] for i in val_idx.indices]

        else:
            train_dataset = datasets.ImageFolder(
                train_path, transform=self.train_transforms
            )
            val_dataset = datasets.ImageFolder(
                val_path, transform=self.val_transforms
            )

        #  Dataloaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )

        # Model: EfficientNet-B0
        self.model = efficientnet_b0(pretrained=True)

        # Freeze backbone
        for param in self.model.features.parameters():
            param.requires_grad = False

        # Replace classifier
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

        # Fine-tune last block + classifier
        for name, param in self.model.named_parameters():
            if "features.6" in name or "classifier" in name:
                param.requires_grad = True

        self.model.to(device)

        # Loss, Optimizer, Scheduler
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=1e-4
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.3, patience=3
        )

        # Checkpoints
        self.best_accuracy = 0.0
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        if os.path.exists(best_path):
            self.model.load_state_dict(torch.load(best_path, map_location=device))
            print("[INFO] Loaded existing best model")

    # Training
    def train(self, epochs: int = 30):
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for images, labels in self.train_loader:
                images, labels = images.to(device), labels.to(device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)
            val_acc = self.evaluate()

            self.scheduler.step(val_acc)

            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%"
            )

            # Save best model
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.checkpoint_dir, "best_model.pth")
                )
                print(f"[INFO] Best model updated ({val_acc:.2f}%)")

    # Evaluation
    def evaluate(self):
        self.model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        return 100.0 * correct / total

    # Inference
    def predict_image(self, image_path: str):
        self.model.eval()
        image = Image.open(image_path).convert("RGB")
        image = self.val_transforms(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = self.model(image)
            _, pred = torch.max(outputs, 1)

        return self.class_names[pred.item()]


trainer = HDAImgTrainer(
    train_path="/kaggle/working/train",
    val_path="/kaggle/working/val",   
    batch_size=32
)

trainer.train(epochs=10)