import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from scipy.io.wavfile import write

# 自定义数据集类
class TTSDataset(Dataset):
    def __init__(self, audio_dir, text_dir):
        self.audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir)]
        self.text_files = [os.path.join(text_dir, f) for f in os.listdir(text_dir)]

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        text_path = self.text_files[idx]
        audio = torch.load(audio_path)
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return audio, text

# TTS 模型类
class TTSModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(TTSModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, text):
        embedded = self.embedding(text)
        output, _ = self.lstm(embedded)
        output = self.fc(output[:, -1, :])
        return output

    def generate(self, input_ids):
        embedded = self.embedding(input_ids)
        output, _ = self.lstm(embedded, self.init_hidden(1))
        audio = output.squeeze().detach().cpu()
        return audio

    def init_hidden(self, batch_size):
        return (torch.zeros(2, batch_size, self.hidden_dim),
                torch.zeros(2, batch_size, self.hidden_dim))

# 训练代码
audio_dir = '/mode/tts/audios'
text_dir = '/mode/tts/texts'
dataset = TTSDataset(audio_dir, text_dir)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = TTSModel(vocab_size=len(dataset.text_files), embedding_dim=128, hidden_dim=256, num_layers=2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(10):
    for audios, texts in dataloader:
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, audios)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# 生成自定义语音
text1 = '这是第一段测试文本'
text2 = '这是第二段测试文本'
audio1 = model.generate(torch.tensor([dataset.encode_text(text1)]))
audio2 = model.generate(torch.tensor([dataset.encode_text(text2)]))

write('custom_audio1.wav', 16000, audio1.numpy())
write('custom_audio2.wav', 16000, audio2.numpy())
