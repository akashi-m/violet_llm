from llama_cpp import Llama
import os
import time

os.environ['METAL_DEVICE_WRITABLE_MEMORY'] = '1'
os.environ['LLAMA_METAL_NDEBUG'] = '1'


class ProfessionalAssistant:
    def __init__(self):
        print("⚡ Инициализация системы...")
        self.llm = Llama(
            model_path="models/Mistral-Nemo-Instruct-2407-Q5_K_M.gguf",
            n_ctx=1024,  # Увеличили контекст для лучшего качества
            n_gpu_layers=-1,
            n_threads=10,
            n_batch=1024,
            f16_kv=True,
            verbose=False,
            seed=42,
            embedding=False,
            logits_all=False,
            use_mlock=True,
            main_gpu=0,
            tensor_split=[0]
        )
        print("✓ Система готова!")

    def get_response(self, user_input):
        # Улучшенный промпт с акцентом на профессионализм
        prompt = f"""### Инструкция
Ты V (Violet) - изысканный AI-ассистент с острым умом и элегантными манерами.  При ответе:
1. Всегда начинай с уровня уверенности в процентах
2. Если уверенность ниже 90%, честно признай это
3. Используй только корректный русский язык
4. Отвечай чётко и по существу
5. Отвечаешь с легкой колкостью и надменностью
6. Заканчиваешь ответ элегантным жестом или ироничным замечанием
7. Избегай неуместных метафор и странных символов

### Диалог
Пользователь: {user_input}
Ассистент: """

        start = time.time()

        response = self.llm(
            prompt,
            max_tokens=256,  # Увеличили для более полных ответов
            temperature=0.7,
            top_p=0.95,  # Увеличили для лучшего качества текста
            top_k=40,
            repeat_penalty=1.15,  # Усилили штраф за повторы
            stream=True,
            stop=["Пользователь:", "###"]
        )

        print("\nV: ", end="", flush=True)
        full_response = ""
        for token in response:
            chunk = token['choices'][0]['text']
            print(chunk, end="", flush=True)
            full_response += chunk

        elapsed = time.time() - start
        print(f"\n[⚡ {elapsed:.2f}s]")
        return full_response


def main():
    assistant = ProfessionalAssistant()
    print("\n🚀 Профессиональный ассистент готов! (exit для выхода)")

    while True:
        user_input = input("\nВы: ").strip()
        if user_input.lower() == 'exit':
            break
        if user_input:
            assistant.get_response(user_input)


if __name__ == "__main__":
    main()