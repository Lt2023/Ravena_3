# Ravena_5

## 🚩项目简介

本项目实现了一个基于 Transformer 架构的中文对话生成模型。该模型结合了多层 Transformer、位置编码、注意力机制以及自定义数据增强技术，能够对用户输入的中文问题生成合适的回答。项目主要基于 TensorFlow 和 Keras，使用了 Jieba 分词和自定义数据增强方法，旨在提供高效、可定制的对话系统解决方案。

## ❔主要特点

- **多层 Transformer 架构**：采用 120 层多头自注意力机制（Multi-Head Attention）和层归一化（Layer Normalization），显著提升模型的上下文理解能力。
- **自定义位置编码**：通过 Positional Encoding 对输入的文本进行位置编码，保留序列中词语的位置关系。
- **数据增强**：对训练数据进行增强处理，包括随机打乱词语顺序和插入/删除词语，以增加模型的泛化能力。
- **中文分词支持**：通过 Jieba 分词工具对中文输入进行分词处理，保证模型在处理中文文本时的准确性。
- **动态学习率调整**：使用自定义学习率调度器（Warm-up 和指数衰减），提高训练稳定性和效果。
- **提前停止与模型检查点**：通过早停（EarlyStopping）和模型检查点（ModelCheckpoint）回调函数，在验证集上监控训练进度，并保存最优模型。
- **GPU 支持**：自动检测可用的 GPU 资源，并启用 TensorFlow 的动态显存分配，支持在 GPU 上进行训练和推理，从而加速模型的训练过程。

## 🖥️功能特点

- **模型训练与优化**：支持训练过程中动态调整学习率，自动保存最佳模型并避免过拟合。
- **对话生成**：通过输入文本，模型能够生成上下文相关的自然语言回答。支持使用温度和 top-p 采样方法来控制生成文本的多样性。
- **Tokenizer 保存与加载**：支持保存和加载 Tokenizer，使得模型可以在不同环境下共享或重用训练数据。
- **文本清理与格式化**：在生成回答时，自动进行文本清理和格式化，保证输出文本的质量。
- **数据加载与增强**：从 JSON 格式的文件中加载训练数据，并对数据进行分词和增强处理，提升模型训练效果。
- **GPU 加速**：通过 `tf.config` 检测 GPU 是否可用，并启用动态显存分配，减少训练过程中的显存占用，提升训练速度。
- **自定义近义词替换**：在生成文本时，支持对常见的词汇进行近义词替换，提升回答的多样性和自然性。


## ⚠️版权声明

- **非商业性使用**：作品可以被共享、复制和修改，但仅限于非商业用途。
- **禁止商业用途**：作品不得用于商业目的，如销售、租赁或任何形式的商业经营。
- **署名要求**：在使用作品时，必须适当地标明作者并提供协议副本。
- **免责声明**：作品是“按原样”提供的，作者不承担因使用作品而产生的任何责任。

## 🫥环境要求

### 📦必要依赖
#### Linux系统即可
- tensorflow
- flask
- pydot
- graphviz
- jieba
- numpy
- termcolor
- tqdm
- keras
- h5py
- flask-cors
- requests
- scikit-learn

### 🥽安装依赖

您可以通过以下命令安装所有依赖项：
```bash
pip install -r requirements.txt
```
## 🧾 数据格式

项目支持自定义数据集，数据需以 JSON 格式存储，每个数据项包含问题（`question`）和回答（`answer`）。示例格式如下：

```json
{
  "data": [
    {
      "question": "你是谁？",
      "answer": "我是一个语言模型。"
    },
    {
      "question": "今天的天气如何？",
      "answer": "今天天气晴，适合出门。"
    }
  ]
}
```

## 🗝️ 修改数据集
打开 train_data.json 文件。
添加新的问题和答案，确保每对数据包含 question 和 answer 字段。
保存文件并重新运行训练脚本。
根据需要，您可以扩展数据集，添加更多的问答对，适应不同领域的需求。
这个版本更简洁明了，直接说明了如何修改数据集和保存文件。

## 😅如何训练模型
准备数据：修改 train_data.json 文件，确保数据格式正确。
训练模型：运行 train.py 文件来开始训练，默认情况下，模型会进行 500 轮训练。如果您希望修改训练轮数，可以调整 model.py 中的 ```patience=500```
交互使用：训练完成后，程序会进入一个交互模式，您可以输入问题并接收模型生成的回答。输入 exit 可退出程序。生成回答当模型训练完成后，您可以通过交互模式提问，模型会返回生成的回答。生成的回答将被逐步显示，每个部分都会用 "回答1", "回答2" 等标号表示，直到回答完整为止。


## 🧑‍🏫 如何微调模型

微调模型的步骤如下：
微调（Fine-Tuning）是指在已有模型的基础上，使用新的数据集或更小的学习率进行训练。对于这个项目，微调模型可以通过以下几种方式进行：

### 1. 准备微调数据集：
微调模型需要您提供一个新的数据集，这个数据集应包含新的问答对，并且格式必须与原始训练数据格式相同（question 和 answer 字段）。请参考上面提供的数据格式来准备您的数据集。

### 2. 加载已有模型：
在微调之前，我们需要加载已经训练好的模型。在 `train.py` 中，您可以指定已有模型文件进行加载。例如，如果您已经训练过模型并且保存了 `Ravena-LLM_Model.keras` 文件，可以直接加载该模型。

```python
model = LanguageModel(model_file='model/Ravena-LLM_Model.keras')
```
### 3. 微调参数设置：
微调时，您可以调整一些参数，特别是 `epochs` 和 `learning_rate`。通常，微调时我们会使用较小的学习率，例如 0.0001，这样可以避免模型发生剧烈的更新。您可以在 `train.py` 中进行调整。

```python
model.train(epochs=20, batch_size=32)  # 例如将训练轮数设置为 20
```
### 4. 训练微调模型：
运行 `train.py` 文件开始微调。微调过程中，模型会根据新的数据集继续训练，但不会像初始训练那样完全改变模型结构，而是根据新的数据进行适应性学习。

### 5. 保存微调后的模型：
微调完成后，您可以将新的模型保存到文件中，便于之后使用。
```python
model.model.save('model/Ravena-LLM_Model_Finetuned.keras')
```
### 6. 微调后的效果：
微调的目的是让模型更好地适应特定领域的数据，而不是从零开始训练。因此，微调通常比从头训练节省时间，并且能取得更好的效果。

在模型微调完成后，您可以继续通过 `run.py`进行交互，测试微调后的模型是否能够更好地回答问题。
## 🫠项目结构
```bash
├── model.py              # 模型定义文件，包含 Transformer 网络和训练代码
├── run.py                # 运行脚本
├── LICENSE               # 许可证
├── model_accuracy_20241115_214946_ea6085ad.png  #模型质量图
├── train.py              # 训练脚本
├── train_data.json       # 训练数据文件（包含 question 和 answer）
├── test.py               # 生成图片文件主要用于,查看模型性能
├── api.py                # 模型API
├── testapi.py            # 测试API
├── fine_tune.py          # 微调脚本
├── Ravena-WebUI.html     # 模型前端界面要求运行API.py
├── requirements.txt      # 项目依赖文件
├── Ravena_tokenizer_output.txt  # 序列Tokenizer对象可查看的版本通过test.py生成
├── Tokenizer.py          # 生成可查看的 Tokenizer 
├── /visualize model
  ├── model_structure.png # 生成后模型框架图
  └── visualize_model.py  #生成脚本
├── /weights
  ├── model_weights.png   # 生成后权重图
  ├── 生成权重.py          # 生成权重可视化文件
  └── model_weights.txt   #权重可视化文本
├── /img-model
  ├── image_model_inference.py # 模型推理
  ├── image_model.py          # 模型定义文件
├──tts
  ├── tts_run.py             # 运行脚本
  └── tts_train.py          # 训练脚本
├── /model
  ├── Ravena-LLM_Model.keras # 训练好的模型权重文件
  └── tokenizer.json         # 用于将文本转化为数字序列的 Tokenizer 对象
└── README.md                # 项目目录以及内容相同，支持英文对话，参数量更大。
```

## 贡献
### 如果您有改进建议或想要贡献代码，欢迎通过 GitHub 提交 issues 或 pull requests。
### 通过 GitHub Issues 提交问题
### 通过 Email:anan15919991635@163.com 联系我


## 😁相关论文和文章
#### 《Sequence to Sequence Learning with Neural Networks》 - Google 深度学习团队提出的 Seq2Seq 模型。
#### 《Long Short-Term Memory》 - LSTM 网络的原始论文，介绍了 LSTM 如何解决传统 RNN 的长程依赖问题。
#### 《A Survey on Language Models》 - 综述文章，介绍了现代语言模型的进展，包括基于 Transformer 的模型（如 GPT 和 BERT）等。