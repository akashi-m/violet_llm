from llama_cpp import Llama
import time
import json
from datetime import datetime
import os
from typing import Dict, List


class VAssistant:
    def __init__(self, model_path: str):
        self.llm = Llama(
            model_path=model_path,
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
        self.log_file = "dialogues.json"
        self.dialogues: List[Dict] = []
        self._load_dialogues()

    def _load_dialogues(self) -> None:
        """Загружает диалоги из файла"""
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    self.dialogues = json.load(f)
        except Exception as e:
            print(f"Ошибка загрузки диалогов: {e}")
            self.dialogues = []

    def _save_dialogues(self) -> None:
        """Сохраняет диалоги в файл"""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.dialogues, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения диалогов: {e}")

    def get_response(self, user_input: str) -> str:
        """Генерирует ответ и логирует диалог"""
        start_time = time.time()

        prompt = f"""### Система
Ты V (Violet) - изысканный AI-ассистент с острым умом и строгими манерами. При ответе:
1. Если уверенность ниже 90%, честно признай это
2. Используй только корректный русский язык
3. Отвечай чётко и по существу
4. Отвечаешь с легкой колкостью и надменностью
5. Заканчиваешь ответ сарказмом или ироничным замечанием
6. Избегай неуместных метафор и странных символов

### Диалог
Пользователь: {user_input}
Ассистент:"""

        # Генерация ответа
        response = self.llm(
            prompt,
            max_tokens=256,
            temperature=0.8,
            top_p=0.95,
            top_k=40,
            repeat_penalty=1.15,
            stream=True,
            stop=["Пользователь:", "###"]
        )

        # Потоковый вывод
        print("\nV: ", end="", flush=True)
        full_response = ""
        for token in response:
            chunk = token['choices'][0]['text']
            print(chunk, end="", flush=True)
            full_response += chunk

        # Подготовка данных в формате, удобном для MongoDB
        elapsed_time = time.time() - start_time
        dialogue_entry = {
            "timestamp": datetime.now().isoformat(),
            "input": user_input,
            "response": full_response,
            "metrics": {
                "response_time": round(elapsed_time, 2),
                "tokens": len(full_response)
            }
        }

        # Сохранение диалога
        self.dialogues.append(dialogue_entry)
        self._save_dialogues()

        print(f"\n[⚡ {elapsed_time:.2f}s]")
        return full_response


def main():
    model_path = "models/Mistral-Nemo-Instruct-2407-Q5_K_M.gguf"
    assistant = VAssistant(model_path)
    print("\n🎩 V к вашим услугам! (exit для выхода)")

    while True:
        user_input = input("\nВы: ").strip()
        if user_input.lower() == 'exit':
            print("До свидания!")
            break
        if user_input:
            assistant.get_response(user_input)

main()
