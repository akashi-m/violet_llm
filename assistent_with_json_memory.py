from llama_cpp import Llama
import os
import time


class PerfectVAssistant:
    def __init__(self):
        print("⚡ Инициализация системы с памятью...")
        self.llm = Llama(
            model_path="models/Mistral-Nemo-Instruct-2407-Q5_K_M.gguf",
            n_ctx=2048,
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
        self.context = []
        print("✓ Система готова!")

    def get_response(self, user_input: str) -> str:
        # Добавляем ввод пользователя в контекст
        self.context.append({"role": "user", "content": user_input})

        # Формируем историю диалога
        dialogue_history = "\n".join([
            f"{'Пользователь' if msg['role'] == 'user' else 'Ассистент'}: {msg['content']}"
            for msg in self.context[-6:]  # Последние 3 обмена (6 сообщений)
        ])

        prompt = f"""### Система
Ты V (Violet) - изысканный AI-ассистент с острым умом и элегантными манерами. При ответе:
1. Всегда начинай с уровня уверенности в процентах
2. Если уверенность ниже 90%, честно признай это
3. Используй только корректный русский язык
4. Отвечай чётко и по существу
5. Отвечаешь с легкой колкостью и надменностью
6. Заканчиваешь ответ элегантным жестом или ироничным замечанием
7. Избегай неуместных метафор и странных символов

### История диалога
{dialogue_history}

### Текущий диалог
Пользователь: {user_input}
Ассистент:"""

        start = time.time()

        response = self.llm(
            prompt,
            max_tokens=256,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            repeat_penalty=1.15,
            stream=True,
            stop=["Пользователь:", "###"]
        )

        print("\nV: ", end="", flush=True)
        full_response = ""
        for token in response:
            chunk = token['choices'][0]['text']
            print(chunk, end="", flush=True)
            full_response += chunk

        # Добавляем ответ в контекст
        self.context.append({"role": "assistant", "content": full_response})

        # Ограничиваем размер контекста
        if len(self.context) > 10:  # Храним последние 5 обменов
            self.context = self.context[-10:]

        elapsed = time.time() - start
        print(f"\n[⚡ {elapsed:.2f}s]")
        return full_response


def main():
    assistant = PerfectVAssistant()
    print("\n🎩 V к вашим услугам! (exit для выхода)")

    while True:
        user_input = input("\nВы: ").strip()
        if user_input.lower() == 'exit':
            print("До свидания! *элегантно раскланивается*")
            break
        if user_input:
            assistant.get_response(user_input)


if __name__ == "__main__":
    main()