from model import LanguageModel

def main():
    model = LanguageModel()  # 初始化模型

    print("加载训练好的模型...")

    while True:
        question = input("你好！请问你有什么问题？输入 'exit' 退出程序。\n")

        if question.lower() == 'exit':  # 如果用户输入 'exit'，则退出
            print("退出程序...")
            break

        response = model.generate_answer(question)  # 获取模型回答
        print(f"模型的回答：{response}\n")

if __name__ == "__main__":
    main()