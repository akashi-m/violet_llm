from llama_cpp import Llama
import os
import time
import json
from datetime import datetime
from typing import Dict, Optional
import uuid


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
        self.logs_dir = "chat_logs"
        self._ensure_logs_directory()

    def _ensure_logs_directory(self) -> None:
        """Создает директорию для логов, если она не существует"""
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)

    def _generate_chat_id(self) -> str:
        """Генерирует уникальный ID для диалога"""
        return f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

    def _save_dialogue(self, chat_data: Dict) -> None:
        """Асинхронно сохраняет диалог в отдельный файл"""
        filename = f"{self.logs_dir}/{chat_data['id']}.json"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(chat_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения диалога: {e}")

    def _get_chat_history(self, chat_id: Optional[str] = None) -> Dict:
        """Получает историю конкретного чата"""
        if not chat_id:
            return {}

        try:
            filename = f"{self.logs_dir}/{chat_id}.json"
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Ошибка загрузки истории: {e}")
        return {}

    def get_response(self, user_input: str) -> str:
        """Генерирует ответ и логирует диалог"""
        start_time = time.time()

        prompt = f"""### Система
Ты V (Violet) - изысканный AI-ассистент с острым умом и элегантными манерами. При ответе:
1. Если уверенность ниже 90%, честно признай это
2. Используй только корректный русский язык
3. Отвечай чётко и по существу
4. Отвечаешь с легкой колкостью и надменностью
5. Заканчиваешь ответ элегантным жестом или ироничным замечанием
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

        # Подготовка данных для логирования
        elapsed_time = time.time() - start_time
        chat_data = {
            "id": self._generate_chat_id(),
            "timestamp": datetime.now().isoformat(),
            "dialogue": {
                "input": user_input,
                "response": full_response,
                "metrics": {
                    "response_time": round(elapsed_time, 2),
                    "prompt_tokens": len(prompt),
                    "response_tokens": len(full_response)
                }
            }
        }

        # Асинхронное сохранение
        self._save_dialogue(chat_data)

        print(f"\n[⚡ {elapsed_time:.2f}s]")
        return full_response


def main():
    model_path = "models/Mistral-Nemo-Instruct-2407-Q5_K_M.gguf"
    assistant = VAssistant(model_path)
    print("\n🎩 V к вашим услугам! (exit для выхода)")

    while True:
        user_input = input("\nВы: ").strip()
        if user_input.lower() == 'exit':
            print("До свидания! *элегантно раскланивается*")
            break
        if user_input:
            assistant.get_response(user_input)


main()