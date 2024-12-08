import torch
import torch.nn as nn
import os
from pydub import AudioSegment
from tqdm import tqdm

# 自定义 Tokenizer 类
class CustomTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.char_to_id = {}
        self.id_to_char = {}
        self.build_vocab()

    def build_vocab(self):
        # 构建字符到索引的映射
        chars = list(set("你好,这是一段测试文本。"))
        for i, char in enumerate(chars):
            self.char_to_id[char] = i
            self.id_to_char[i] = char

    def encode(self, text):
        # 将文本编码为索引序列
        return [self.char_to_id[char] for char in text]

    def decode(self, ids):
        # 将索引序列解码为文本
        return ''.join([self.id_to_char[id] for id in ids if id in self.id_to_char])

# TTS 模型
class TTSModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(TTSModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8), num_layers=num_layers)
        self.transformer_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=8), num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        encoder_output = self.transformer_encoder(embedded)
        decoder_output = self.transformer_decoder(embedded, encoder_output)
        output = self.fc(decoder_output)
        return output

# 运行代码
tokenizer = CustomTokenizer(vocab_size=len(set("你好,这是一段测试文本。")))
tts_model = TTSModel(vocab_size=tokenizer.vocab_size, embedding_dim=256, hidden_dim=512, num_layers=6)
tts_model.to(device='cpu')
tts_model.load_state_dict(torch.load('model/tts_model.pth', map_location='cpu'))

# 让用户输入文本
text = input("Enter the text to be converted to speech: ")

# 生成语音
input_ids = torch.tensor([tokenizer.encode(text)], device='cpu')
output = tts_model(input_ids)
audio = tokenizer.decode(output.argmax(dim=-1).squeeze())

print(f"Audio: {audio}")
