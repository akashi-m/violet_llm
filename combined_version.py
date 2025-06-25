import io
import threading
import queue
import time
import wave

import numpy as np
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Any, Dict
from threading import Lock

from deepgram import DeepgramClient
from deepgram.clients.listen.v1 import PrerecordedOptions
from dotenv import load_dotenv
from tts_deepgram import TTSPlayer
from voice_process_classes import VoiceAnalyzer, VoiceActivityDetector
import os
from assistent_with_json_memory import VAssistant

load_dotenv()


class AssistantState(Enum):
    WAITING = auto()  # Ожидание активации
    LISTENING = auto()  # Активное слушание команды
    PROCESSING = auto()  # Обработка команды
    SPEAKING = auto()  # Воспроизведение ответа
    SHUTDOWN = auto()  # Состояние завершения работы


@dataclass
class Message:
    type: str
    data: Any
    confidence: float = 1.0


class VoiceAssistantCore:
    def __init__(self):
        # Базовые параметры и таймауты
        self.COMMAND_TIMEOUT = 5.0
        self.SILENCE_TIMEOUT = 2.0
        self.MIN_COMMAND_DURATION = 0.5
        self.CALIBRATION_DURATION = 2.0  # Длительность калибровки в секундах

        # Таймеры состояний
        self.last_activity_time = time.time()
        self.command_start_time = None
        self.silence_start_time = None

        # Блокировки для потокобезопасности
        self.metrics_lock = Lock()
        self.state_lock = Lock()
        self.shutdown_event = threading.Event()

        # Метрики с защитой от гонок
        self._metrics = {
            'activations': 0,
            'commands_processed': 0,
            'recognition_failures': 0,
            'avg_command_duration': 0.0,
            'total_uptime': 0.0
        }

        self.initialized = False  # Начальное состояние
        try:
            self._init_components()
        except Exception as e:
            print(f"[КРИТИЧЕСКАЯ ОШИБКА] Ошибка инициализации: {e}")

    def _init_components(self):
        """Инициализация всех компонентов системы"""
        try:
            # Очереди для межпоточного взаимодействия
            self.audio_queue = queue.Queue()
            self.command_queue = queue.Queue()
            self.response_queue = queue.Queue()
            self.tts_queue = queue.Queue()

            # Установка начального состояния
            self._state = AssistantState.WAITING

            # Загрузка голосового профиля
            print("[СИСТЕМА] Инициализация анализатора голоса...")
            self.voice_analyzer = VoiceAnalyzer()
            if not self.voice_analyzer.load_profile("models/voice_profile.pkl"):
                raise RuntimeError("Не удалось загрузить голосовой профиль")

            # Инициализация VAD
            print("[СИСТЕМА] Инициализация детектора голосовой активности...")
            self.vad = VoiceActivityDetector(
                rate=48000,
                chunk_size=1024
            )

            # Инициализация LLM
            print("[СИСТЕМА] Инициализация языковой модели...")
            model_path = "models/Mistral-Nemo-Instruct-2407-Q5_K_M.gguf"
            self.llm = VAssistant(model_path)

            # Инициализация TTS
            print("[СИСТЕМА] Инициализация синтезатора речи...")
            self.tts = TTSPlayer()

            # Предзагруженные фразы для быстрых ответов
            self.quick_responses = {
                "activation": "Слушаю",
                "thinking": "Дайте подумать",
                "searching": "Ищу информацию",
                "processing": "Обрабатываю запрос",
                "error": "Произошла ошибка, повторите запрос",
                "shutdown": "Завершаю работу"
            }

            self.initialized = True  # Перемещаем это в конец метода
            print("[СИСТЕМА] Инициализация компонентов завершена успешно")
            return True  # Добавляем возврат успешной инициализации


        except Exception as e:
            print(f"[КРИТИЧЕСКАЯ ОШИБКА] Ошибка инициализации: {e}")
            self.initialized = False  # Явно устанавливаем False при ошибке
            self.shutdown()
            raise

    @property
    def state(self) -> AssistantState:
        with self.state_lock:
            return self._state

    @state.setter
    def state(self, new_state: AssistantState):
        """Установка состояния с логированием перехода"""
        with self.state_lock:
            old_state = self._state
            self._state = new_state

            # Логирование перехода состояния
            print(f"[СИСТЕМА] Переход состояния: {old_state.name} -> {new_state.name}")
            self.last_activity_time = time.time()

            # Обработка специфичных действий для каждого состояния
            if new_state == AssistantState.LISTENING:
                self.command_start_time = time.time()
                self.silence_start_time = None
            elif new_state == AssistantState.PROCESSING:
                if self.command_start_time:
                    duration = time.time() - self.command_start_time
                    with self.metrics_lock:
                        self._metrics['commands_processed'] += 1
                        n = self._metrics['commands_processed']
                        current_avg = self._metrics['avg_command_duration']
                        self._metrics['avg_command_duration'] = (current_avg * (n - 1) + duration) / n
            elif new_state == AssistantState.WAITING:
                self.command_start_time = None
                self.silence_start_time = None

    def audio_capture_thread(self):
        """Поток захвата и предобработки аудио"""
        print("[СИСТЕМА] Запуск потока захвата аудио")

        # Калибровка VAD перед началом работы
        if not self.vad.calibrated:
            print("[СИСТЕМА] Калибровка уровня шума...")
            if not self.vad.calibrate(self.CALIBRATION_DURATION):
                print("[ОШИБКА] Не удалось выполнить калибровку")
                self.shutdown()
                return

        while not self.shutdown_event.is_set():
            try:
                # Захват аудио фрейма
                audio_frame = self.vad.get_audio_frame()

                if audio_frame is not None:
                    # Проверка голосовой активности
                    if self.vad.is_speech(audio_frame):
                        self.audio_queue.put(Message("audio", audio_frame))

                time.sleep(0.01)  # Небольшая задержка для снижения нагрузки

            except Exception as e:
                print(f"[ОШИБКА] Поток захвата аудио: {e}")
                if str(e).lower().find("device") != -1:  # Проблемы с устройством
                    self.shutdown()
                    break

    def voice_analysis_thread(self):
        """Поток анализа голоса и активации"""
        print("[СИСТЕМА] Запуск потока анализа голоса")

        while not self.shutdown_event.is_set():
            try:
                # Проверка таймаутов
                self._check_timeouts()

                # Получение аудио из очереди
                try:
                    msg = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if self.state == AssistantState.WAITING:
                    # Проверка на активационное слово и голос
                    similarity = self.voice_analyzer.compare_voice(msg.data)

                    if similarity >= self.voice_analyzer.confidence_threshold:
                        old_state = self.state
                        self.state = AssistantState.LISTENING
                        print(f"[СИСТЕМА] Переход состояния: {old_state.name} -> {self.state.name}")
                        self.last_activity_time = time.time()

                        if self.state == AssistantState.LISTENING:
                            self.command_start_time = time.time()
                            self.silence_start_time = None

                        with self.metrics_lock:
                            self._metrics['activations'] += 1

                        self.tts_queue.put(Message("speak", self.quick_responses["activation"]))

                elif self.state == AssistantState.LISTENING:
                    # Проверка на тишину
                    if not self.vad.is_speech(msg.data):
                        if self.silence_start_time is None:
                            self.silence_start_time = time.time()
                    else:
                        self.silence_start_time = None
                        self.last_activity_time = time.time()
                        self.command_queue.put(msg)

                self.audio_queue.task_done()

            except Exception as e:
                print(f"[ОШИБКА] Поток анализа голоса: {e}")
                with self.metrics_lock:
                    self._metrics['recognition_failures'] += 1

    def command_processing_thread(self):
        """Поток обработки команд через LLM"""
        print("[СИСТЕМА] Запуск потока обработки команд")

        while not self.shutdown_event.is_set():
            try:
                if self.state == AssistantState.PROCESSING:
                    # Получение команды
                    command = ""
                    try:
                        while not self.command_queue.empty():
                            msg = self.command_queue.get_nowait()
                            text = self._process_audio_to_text(msg.data)
                            if text:
                                command += " " + text
                            self.command_queue.task_done()
                    except queue.Empty:
                        pass

                    if command.strip():
                        # Проверка на команду завершения
                        if any(phrase in command.lower() for phrase in ["отключайся", "выключайся", "завершай работу"]):
                            self.tts_queue.put(Message("speak", self.quick_responses["shutdown"]))
                            self.shutdown()
                            break

                        # Быстрый ответ для обратной связи
                        self.tts_queue.put(Message("speak", self.quick_responses["thinking"]))

                        # Обработка через LLM
                        response = self.llm.get_response(command)
                        if response:
                            self.response_queue.put(Message("response", response))
                            self.state = AssistantState.SPEAKING
                        else:
                            self.tts_queue.put(Message("speak", self.quick_responses["error"]))
                            self.state = AssistantState.WAITING
                    else:
                        # Если команда пустая, возвращаемся в режим ожидания
                        self.state = AssistantState.WAITING

                time.sleep(0.1)

            except Exception as e:
                print(f"[ОШИБКА] Поток обработки команд: {e}")
                self.tts_queue.put(Message("speak", self.quick_responses["error"]))
                self.state = AssistantState.WAITING

    def tts_thread(self):
        """Поток синтеза и воспроизведения речи"""
        print("[СИСТЕМА] Запуск потока синтеза речи")

        while not self.shutdown_event.is_set():
            try:
                # Приоритет отдается очереди TTS
                try:
                    msg = self.tts_queue.get(timeout=0.1)
                    if msg.type == "speak":
                        self.tts.say(msg.data)
                    self.tts_queue.task_done()
                except queue.Empty:
                    # Если в очереди TTS пусто, проверяем очередь ответов
                    if self.state == AssistantState.SPEAKING:
                        try:
                            msg = self.response_queue.get_nowait()
                            if msg.type == "response":
                                self.tts.say(msg.data)
                            self.response_queue.task_done()
                            self.state = AssistantState.WAITING
                        except queue.Empty:
                            pass

            except Exception as e:
                print(f"[ОШИБКА] Поток синтеза речи: {e}")
                self.state = AssistantState.WAITING

    def start(self):
        """Запуск всех компонентов системы"""
        if not self.initialized:
            print("[ОШИБКА] Система не инициализирована")
            return

        try:
            # Инициализация потоков
            self.threads = {
                "audio": threading.Thread(target=self.audio_capture_thread),
                "voice": threading.Thread(target=self.voice_analysis_thread),
                "command": threading.Thread(target=self.command_processing_thread),
                "tts": threading.Thread(target=self.tts_thread)
            }

            # Запуск потоков
            for name, thread in self.threads.items():
                thread.daemon = True
                thread.start()
                print(f"[СИСТЕМА] Поток {name} запущен")

            print("[СИСТЕМА] Ассистент готов к работе")

            # Основной цикл
            start_time = time.time()
            while not self.shutdown_event.is_set():
                self._metrics['total_uptime'] = time.time() - start_time
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n[СИСТЕМА] Получен сигнал завершения")
            self.shutdown()

        except Exception as e:
            print(f"[ОШИБКА] Основной поток: {e}")
            self.shutdown()

        finally:
            self._print_final_metrics()

    def shutdown(self):
        """Корректное завершение работы системы"""
        if not self.shutdown_event.is_set():
            print("[СИСТЕМА] Начало процедуры завершения...")
            self.shutdown_event.set()
            self.state = AssistantState.SHUTDOWN

            # Очистка очередей
            for q in [self.audio_queue, self.command_queue,
                      self.response_queue, self.tts_queue]:
                while not q.empty():
                    try:
                        q.get_nowait()
                        q.task_done()
                    except queue.Empty:
                        break

            # Ожидание завершения потоков
            if hasattr(self, 'threads'):
                for name, thread in self.threads.items():
                    thread.join(timeout=2.0)
                    print(f"[СИСТЕМА] Поток {name} остановлен")

            # Освобождение ресурсов
            if hasattr(self, 'vad'):
                del self.vad
            if hasattr(self, 'tts'):
                del self.tts

            print("[СИСТЕМА] Работа завершена")

    def _check_timeouts(self):
        """Проверка таймаутов и управление состояниями"""
        current_time = time.time()

        if self.state == AssistantState.LISTENING:
            # Проверка длительности тишины
            if self.silence_start_time and (current_time - self.silence_start_time) >= self.SILENCE_TIMEOUT:
                if self.command_start_time and (current_time - self.command_start_time) >= self.MIN_COMMAND_DURATION:
                    self.state = AssistantState.PROCESSING
                else:
                    print("[СИСТЕМА] Команда слишком короткая, возврат в режим ожидания")
                    self.state = AssistantState.WAITING
                return

            # Проверка общего таймаута команды
            if self.command_start_time and (current_time - self.command_start_time) >= self.COMMAND_TIMEOUT:
                print("[СИСТЕМА] Таймаут команды, переход к обработке")
                self.state = AssistantState.PROCESSING
                return

        elif self.state == AssistantState.SPEAKING:
            # Если нет активности после ответа
            if (current_time - self.last_activity_time) >= self.SILENCE_TIMEOUT:
                print("[СИСТЕМА] Возврат в режим ожидания после ответа")
                self.state = AssistantState.WAITING

    def _update_metrics(self, event_type: str, value: float = 1.0):
        """Потокобезопасное обновление метрик"""
        with self.metrics_lock:
            if event_type == 'activation':
                self._metrics['activations'] += value
            elif event_type == 'command':
                self._metrics['commands_processed'] += value
                # Обновление средней длительности команд
                n = self._metrics['commands_processed']
                current_avg = self._metrics['avg_command_duration']
                self._metrics['avg_command_duration'] = (current_avg * (n - 1) + value) / n
            elif event_type == 'recognition_failure':
                self._metrics['recognition_failures'] += value

    def _log_state_transition(self, old_state: AssistantState, new_state: AssistantState):
        """Логирование и обработка переходов между состояниями"""
        print(f"[СИСТЕМА] Переход состояния: {old_state.name} -> {new_state.name}")
        self.last_activity_time = time.time()

        if new_state == AssistantState.LISTENING:
            self.command_start_time = time.time()
            self.silence_start_time = None
        elif new_state == AssistantState.PROCESSING:
            if self.command_start_time:
                duration = time.time() - self.command_start_time
                self._update_metrics('command', duration)
        elif new_state == AssistantState.WAITING:
            self.command_start_time = None
            self.silence_start_time = None

    def _process_audio_to_text(self, audio_data: np.ndarray) -> str:
        """Преобразование аудио в текст через Deepgram API"""
        try:
            print(f"[DEBUG] Размер аудио данных: {len(audio_data)}")
            print(f"[DEBUG] Тип данных: {audio_data.dtype}")
            print(f"[DEBUG] Диапазон значений: {np.min(audio_data)} to {np.max(audio_data)}")

            # Конвертация numpy array в bytes
            byte_buffer = io.BytesIO()
            with wave.open(byte_buffer, 'wb') as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(48000)
                wf.writeframes(audio_data.tobytes())

            audio_data_bytes = byte_buffer.getvalue()
            print(f"[DEBUG] Размер WAV буфера: {len(audio_data_bytes)} bytes")

            # Инициализация Deepgram клиента
            client = DeepgramClient(os.getenv('DEEPGRAM_API_KEY'))
            print("[DEBUG] Deepgram клиент инициализирован")

            # Настройка параметров распознавания
            options = PrerecordedOptions(
                smart_format=True,
                model="nova-2",
                language="ru"
            )

            # Отправка на распознавание
            response = client.listen.prerecorded.v('1').transcribe_file(
                {'buffer': audio_data_bytes},
                options
            )
            print("[DEBUG] Получен ответ от Deepgram")
            print(f"[DEBUG] Ответ: {response}")

            # Извлечение результата
            if (response.results and
                    response.results.channels and
                    response.results.channels[0].alternatives):
                transcript = response.results.channels[0].alternatives[0].transcript
                confidence = response.results.channels[0].alternatives[0].confidence

                print(f"[STT] Уверенность распознавания: {confidence:.2f}")
                print(f"[STT] Распознанный текст: {transcript}")
                return transcript

            print("[DEBUG] Не удалось извлечь результат из ответа")
            return ""

        except Exception as e:
            print(f"[ОШИБКА] Процесс STT: {e}")
            print(f"[DEBUG] Полный traceback: {traceback.format_exc()}")
            return ""

    def _print_final_metrics(self):
        """Вывод финальной статистики работы"""
        with self.metrics_lock:
            print("\n=== Статистика работы ===")
            print(f"Время работы: {self._metrics['total_uptime']:.1f} сек")
            print(f"Активаций: {self._metrics['activations']}")
            print(f"Обработано команд: {self._metrics['commands_processed']}")
            print(f"Ошибок распознавания: {self._metrics['recognition_failures']}")
            if self._metrics['commands_processed'] > 0:
                print(f"Средняя длительность команды: {self._metrics['avg_command_duration']:.1f} сек")
            print("======================")


if __name__ == "__main__":
    assistant = VoiceAssistantCore()
    assistant.start()