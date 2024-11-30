import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import jieba
import torchvision.transforms as transforms

# 自定义数据集类
class ChineseImageDataset(Dataset):
    def __init__(self, image_dir, text_dir, transform=None):
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.transform = transform
        self.image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
        self.text_files = [os.path.join(text_dir, f) for f in os.listdir(text_dir)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        text_path = self.text_files[idx]
        image = Image.open(image_path).convert('RGB')
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        if self.transform:
            image = self.transform(image)
        return image, self.encode_text(text)

    def encode_text(self, text):
        words = jieba.lcut(text)
        vocab = list(set(words))
        word_to_id = {word: i for i, word in enumerate(vocab)}
        encoded_text = [word_to_id[word] for word in words]
        return torch.tensor(encoded_text)

# 自定义图像理解模型类
class ChineseImageUnderstandingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super(ChineseImageUnderstandingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7 + embedding_dim, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, image, text):
        image_features = self.pool2(self.relu2(self.conv2(self.pool1(self.relu1(self.conv1(image))))))
        image_features = image_features.view(-1, 64 * 7 * 7)
        text_features = self.embedding(text).mean(dim=1)
        combined_features = torch.cat((image_features, text_features), dim=1)
        x = self.relu3(self.fc1(combined_features))
        x = self.fc2(x)
        return x

# 训练代码
image_dir = '/mode/image_model/images'
text_dir = '/mode/image_model/texts'
dataset = ChineseImageDataset(image_dir, text_dir, transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

vocab_size = len(dataset.encode_text('').unique())
model = ChineseImageUnderstandingModel(vocab_size=vocab_size, embedding_dim=128, num_classes=10)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for images, texts in dataloader:
        optimizer.zero_grad()
        outputs = model(images, texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# 保存模型
torch.save(model.state_dict(), 'chinese_image_model.pth')

# 预测未知图像
unknown_image = Image.open('path/to/unknown/image.jpg').convert('RGB')
unknown_image = transforms.Resize((224, 224))(unknown_image)
unknown_image = transforms.ToTensor()(unknown_image)
unknown_image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(unknown_image)
unknown_image = unknown_image.unsqueeze(0)

unknown_text = '我不确定这是什么图片'
unknown_text_ids = torch.tensor([dataset.encode_text(unknown_text)], device='cpu')

unknown_output = model(unknown_image, unknown_text_ids)
unknown_predicted_class = torch.argmax(unknown_output, dim=1).item()
print(f'Predicted class for unknown image: {unknown_predicted_class}')
