import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torchvision.transforms as transforms

# 自定义数据集类
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# 自定义图像分类模型类
class ImageClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super(ImageClassificationModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# 训练代码
image_dir = '/mode/image_model/images'
dataset = ImageDataset(image_dir, transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = ImageClassificationModel(num_classes=10)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for images in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# 保存模型
torch.save(model.state_dict(), 'image_classification_model.pth')

# 预测未知图像
unknown_image = Image.open('path/to/unknown/image.jpg').convert('RGB')
unknown_image = transforms.Resize((224, 224))(unknown_image)
unknown_image = transforms.ToTensor()(unknown_image)
unknown_image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(unknown_image)
unknown_image = unknown_image.unsqueeze(0)

unknown_output = model(unknown_image)
unknown_predicted_class = torch.argmax(unknown_output, dim=1).item()
print(f'Predicted class for unknown image: {unknown_predicted_class}')
