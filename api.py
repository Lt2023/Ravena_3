# api.py

from flask import Flask, request, jsonify
from model import LanguageModel
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 启用 CORS 支持

# 初始化语言模型
model = LanguageModel()

@app.route('/ask', methods=['POST'])
def ask():
    # 获取请求中的问题
    data = request.get_json()
    question = data.get('question', '')

    if not question:
        return jsonify({"error": "没有提供问题"}), 400

    # 获取模型的回答
    response = model.generate_answer(question)

    # 返回回答
    return jsonify({"answer": response})

@app.route('/')
def home():
    return "请打开文件目录中的HTML文件在浏览器中对话。"

if __name__ == "__main__":
    print("服务已启动，正在监听端口 5000...")
    app.run(debug=True, host="0.0.0.0", port=5000)
