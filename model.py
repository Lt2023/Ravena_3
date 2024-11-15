import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LayerNormalization, Dropout, Dense, Input, MultiHeadAttention, Layer, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import jieba
import json
from termcolor import colored
from tqdm import tqdm
import os
import random
import re
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint

class PositionalEncoding(Layer):
    def __init__(self, max_len, model_dim):
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len
        self.model_dim = model_dim

    def call(self, inputs):
        position = np.arange(self.max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.model_dim, 2) * -(np.log(10000.0) / self.model_dim))
        pe = np.zeros((self.max_len, self.model_dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[np.newaxis, ...]
        return inputs + tf.cast(pe, tf.float32)

class LanguageModel:
    def __init__(self, vocab_size=10000, max_seq_length=20, data_file='train_data.json', model_file='model/Ravena-LLM_Model.h5', tokenizer_file='model/tokenizer.json'):
        print(colored("初始化模型...", "yellow"))

        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.data_file = data_file
        self.model_file = model_file.replace('.h5', '.keras') 
        self.tokenizer_file = tokenizer_file
        self.tokenizer = None
        self.model = self.build_model()
        self.load_data()
        self.previous_answers = set()
        self.is_trained = False

    def load_data(self):
        try:
            with open(self.data_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
                self.data = data
        except FileNotFoundError:
            print(colored(f"未找到数据文件：{self.data_file}，将创建一个新的文件。", "red"))
            self.data = {"data": []}

        questions = [item['question'] for item in self.data['data']]
        answers = [item['answer'] for item in self.data['data']]

        questions = self.augment_data(questions)
        answers = self.augment_data(answers)

        questions = [" ".join(jieba.cut(q)) for q in questions]
        answers = [" ".join(jieba.cut(a)) for a in answers]

        if not os.path.exists(self.tokenizer_file):
            self.tokenizer = Tokenizer(num_words=self.vocab_size)
            self.tokenizer.fit_on_texts(questions + answers)

            with open(self.tokenizer_file, 'w', encoding='utf-8') as f:
                json.dump(self.tokenizer.to_json(), f, ensure_ascii=False, indent=4)
            print(colored(f"Tokenizer 文件已保存：{self.tokenizer_file}", "green"))
        else:
            with open(self.tokenizer_file, 'r', encoding='utf-8') as f:
                tokenizer_json = json.load(f)
                self.tokenizer = tokenizer_from_json(tokenizer_json)
            print(colored(f"成功加载Tokenizer文件: {self.tokenizer_file}", "green"))

        self.question_sequences = pad_sequences(self.tokenizer.texts_to_sequences(questions), maxlen=self.max_seq_length)
        self.answer_sequences = [self.tokenizer.texts_to_sequences([a])[0] for a in answers]
        self.answer_sequences = np.array([seq[0] for seq in self.answer_sequences])

    def augment_data(self, data):
        """对数据进行更复杂的增强"""
        augmented_data = []
        for item in data:
            words = item.split()
            random.shuffle(words)  

            if random.random() > 0.5:
                words.append(random.choice(words))  
            if len(words) > 2 and random.random() > 0.5:
                words.remove(random.choice(words))  

            augmented_data.append(' '.join(words))
        return augmented_data

    def build_model(self):
        """构建多层 Transformer 模型"""
        print(colored("构建 Transformer 模型...", "yellow"))

        input_layer = Input(shape=(self.max_seq_length,))
        embedding_layer = Embedding(self.vocab_size, 128)(input_layer)

        pos_encoding = PositionalEncoding(self.max_seq_length, 128)(embedding_layer)
        x = pos_encoding
        for _ in range(8):  # 增加 Transformer 层数为 8
            attention = MultiHeadAttention(num_heads=8, key_dim=128)(x, x)
            attention = LayerNormalization()(attention)
            attention = Dropout(0.1)(attention)
            x = attention

        pooling = GlobalAveragePooling1D()(x)
        dropout = Dropout(0.5)(pooling)
        output_layer = Dense(self.vocab_size, activation='softmax')(dropout)

        model = Model(inputs=input_layer, outputs=output_layer)

        # 使用 AdamW 优化器，调整学习率调度
        optimizer = Adam(learning_rate=0.0001)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def scheduler(self, epoch, lr):
        if epoch < 5:
            return float(lr * (epoch + 1) / 5)  # Warm-Up 阶段增加学习率
        else:
            return float(lr * np.exp(-0.1))  # 训练后期衰减学习率

    def train(self, epochs=100, batch_size=64):
        print(colored("开始训练模型...", "yellow"))
        
        # 设置回调函数
        lr_scheduler = LearningRateScheduler(self.scheduler)
        early_stopping = EarlyStopping(monitor='val_loss', patience=500, restore_best_weights=True, verbose=1)  # 提前停止
        model_checkpoint = ModelCheckpoint(self.model_file, monitor='val_loss', save_best_only=True, verbose=1)  # 保存最佳模型
        
        # 使用动态 batch size 和其他参数
        self.model.fit(self.question_sequences, self.answer_sequences, 
                       epochs=epochs, 
                       batch_size=batch_size, 
                       verbose=1, 
                       validation_split=0.1,  # 使用一部分数据作为验证集
                       callbacks=[lr_scheduler, early_stopping, model_checkpoint])

        self.is_trained = True
        print(colored("训练完成！模型已保存。", "yellow"))

    def generate_answer(self, input_text, max_length=20, temperature=0.7, top_p=0.9):
        """生成回答"""
        seq = pad_sequences(self.tokenizer.texts_to_sequences([input_text]), maxlen=self.max_seq_length)
        generated_seq = list(seq[0])

        generated_text = ''

        for _ in range(max_length):
            if len(generated_seq) > self.max_seq_length:
                generated_seq = generated_seq[-self.max_seq_length:]

            pred = self.model.predict(np.array([generated_seq]), verbose=0)
            pred = np.log(pred) / temperature  # 调整温度
            pred = np.exp(pred) / np.sum(np.exp(pred))

            sorted_indices = np.argsort(pred[0])[::-1]
            cumulative_probs = np.cumsum(pred[0][sorted_indices])
            top_p_indices = sorted_indices[cumulative_probs <= top_p]

            next_word_index = np.random.choice(top_p_indices)
            generated_seq.append(next_word_index)

            if next_word_index == 0:
                break

            word = self.tokenizer.index_word.get(next_word_index, '')
            generated_text += word + ' '

        generated_text = self.clean_text(generated_text)
        return self.format_text(generated_text)

    def clean_text(self, text):
        cleaned_text = ' '.join(text.split())
        cleaned_text = cleaned_text.replace("  ", " ").strip()
        return cleaned_text

    def format_text(self, text):
        text = re.sub(r'\s([?.!,":;(){}])', r'\1', text) 
        text = re.sub(r'\n', ' ', text)  
        text = text.strip()
        return text
