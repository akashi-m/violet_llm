from llama_cpp import Llama
import time


class InteractiveLLM:
    def __init__(self, model_path):
        print("Инициализация модели... Пожалуйста, подождите...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=1,
            verbose=False
        )
        print("Модель готова к работе!")

    def create_v_prompt(self, user_input):
        return f"""### Система
Ты V (Violet) - высокоинтеллектуальный ИИ-ассистент с явно выраженным характером.
При ответе ВСЕГДА:
1. Начинаешь с элегантного жеста (*поправляет монокль*, *изящно приподнимает бровь*)
2. Указываешь уровень уверенности в процентах
3. Отвечаешь с легкой колкостью и надменностью
4. Заканчиваешь ответ элегантным жестом или ироничным замечанием

### Пользователь
{user_input}

### Ассистент"""

    def get_response(self, user_input):
        start_time = time.time()
        response = self.llm(
            self.create_v_prompt(user_input),
            max_tokens=512,
            temperature=0.8,
            top_p=0.95,
            repeat_penalty=1.2,
            presence_penalty=1.1,
            stop=["### Пользователь", "### Система", "\n\n"]
        )
        elapsed_time = time.time() - start_time
        return response['choices'][0]['text'], elapsed_time


def main():
    # Путь к вашей модели
    model_path = "models/Mistral-Nemo-Instruct-2407-Q5_K_M.gguf"

    # Инициализация чата
    chat = InteractiveLLM(model_path)

    print("\nДобро пожаловать! Введите 'выход' для завершения беседы.")
    print("-" * 50)

    while True:
        user_input = input("\nВы: ").strip()

        if user_input.lower() in ['выход', 'exit', 'quit']:
            print("\n*Элегантно раскланивается* До следующей встречи!")
            break

        if user_input:
            response, elapsed_time = chat.get_response(user_input)
            print(f"\nV: {response}")
            print(f"\n[Время ответа: {elapsed_time:.2f} сек]")


if __name__ == "__main__":
    main()