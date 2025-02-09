from llama_cpp import Llama
import os
import time

# Оптимизация для Metal
os.environ['METAL_DEVICE_WRITABLE_MEMORY'] = '1'
os.environ['LLAMA_METAL_PATH_OVERRIDE'] = './models'
os.environ['LLAMA_METAL_NDEBUG'] = '1'


class UltraFastAssistant:
    def __init__(self):
        print("⚡ Инициализация Metal Turbo...")
        self.llm = Llama(
            model_path="models/Mistral-Nemo-Instruct-2407-Q5_K_M.gguf",
            n_ctx=512,
            n_gpu_layers=-1,
            n_threads=10,  # Threads задаются здесь, не в вызове
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
        print("✓ Metal Turbo активирован!")

    def get_response(self, user_input):
        prompt = f"""У: {user_input}
О:"""

        start = time.time()

        response = self.llm(
            prompt,
            max_tokens=128,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            stream=True,
            stop=["У:", "О:"],
            mirostat_mode=2,
            mirostat_tau=5.0,
            mirostat_eta=0.1
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
    assistant = UltraFastAssistant()
    print("\n🚀 Metal Turbo готов! (exit для выхода)")

    while True:
        user_input = input("\nВы: ").strip()
        if user_input.lower() == 'exit':
            break
        if user_input:
            assistant.get_response(user_input)


if __name__ == "__main__":
    main()