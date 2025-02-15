import pyaudio
import numpy as np
import asyncio
import time
import os
from enum import Enum
from collections import deque
from scipy.signal import butter, lfilter
import queue
import threading
from typing import Optional, Callable
import io
import wave
from deepgram import DeepgramClient, PrerecordedOptions
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class AssistantState(Enum):
    WAITING_FOR_ACTIVATION = 1
    LISTENING_FOR_COMMAND = 2
    PROCESSING_RESPONSE = 3
    WAITING_FOR_FOLLOWUP = 4


class VoiceActivator:
    def __init__(self,
                 on_activation: Callable[[], None],
                 on_command: Callable[[str], None],
                 on_deactivation: Callable[[], None]):
        # Инициализация параметров
        self.RATE = 44100
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1

        # Колбэки
        self.on_activation = on_activation
        self.on_command = on_command
        self.on_deactivation = on_deactivation

        # VAD параметры
        self.speech_window = deque(maxlen=20)
        self.energy_threshold = None
        self.speech_count = 0
        self.silence_count = 0

        # Состояние системы
        self.state = AssistantState.WAITING_FOR_ACTIVATION
        self.last_activity_time = time.time()
        self.noise_threshold = None
        self.is_calibrated = False

        # Буферы и очереди
        self.audio_queue = queue.Queue()
        self.speech_buffer = []
        self.activation_buffer = deque(maxlen=int(self.RATE * 2))

        # Компоненты
        self.audio_interface = pyaudio.PyAudio()
        self.stream = None
        self.dg_client = DeepgramClient(os.getenv('DEEPGRAM_API_KEY'))
        self.is_running = False

        # Тайминги
        self.SILENCE_LIMIT = 1.2  # Пауза для окончания команды
        self.LONG_SILENCE = 4.0  # Пауза для деактивации
        self.MAX_COMMAND_TIME = 30.0

        # Инициализация фильтров
        self._setup_filters()

    def _setup_filters(self):
        nyq = 0.5 * self.RATE
        low = 85 / nyq
        high = 3000 / nyq
        self.b, self.a = butter(5, [low, high], btype='band')

    def start(self):
        """Запуск системы"""
        if not self.is_calibrated:
            self.calibrate()

        self.is_running = True
        self.stream = self.audio_interface.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            stream_callback=self._audio_callback
        )

        self.stream.start_stream()
        threading.Thread(target=self._process_audio_queue, daemon=True).start()
        print("Система активирована и ожидает команды")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        if self.is_running:
            self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)

    def _process_audio_queue(self):
        """Основной цикл обработки аудио"""
        while self.is_running:
            if self.audio_queue.empty():
                time.sleep(0.01)
                continue

            audio_chunk = self.audio_queue.get()
            current_time = time.time()

            if self.state == AssistantState.WAITING_FOR_ACTIVATION:
                self._handle_activation_mode(audio_chunk)
            elif self.state == AssistantState.LISTENING_FOR_COMMAND:
                self._handle_listening_mode(audio_chunk, current_time)
            elif self.state == AssistantState.WAITING_FOR_FOLLOWUP:
                self._handle_followup_mode(audio_chunk, current_time)

    def _handle_activation_mode(self, audio_chunk):
        """Режим ожидания активации"""
        # Добавляем в буфер для анализа
        if self.is_speech(audio_chunk):
            self.activation_buffer.append(audio_chunk)

            # Проверяем на наличие команды активации
            if len(self.activation_buffer) >= 10:  # примерно 0.5 сек
                audio_data = self._frames_to_audio_data(list(self.activation_buffer))
                self._check_activation(audio_data)  # Теперь синхронный вызов
        else:
            self.activation_buffer.clear()

    def _handle_listening_mode(self, audio_chunk, current_time):
        """Режим прослушивания команды"""
        if self.is_speech(audio_chunk):
            self.speech_buffer.append(audio_chunk)
            self.last_activity_time = current_time
            print("▶", end="", flush=True)
        else:
            print(".", end="", flush=True)
            if current_time - self.last_activity_time > self.SILENCE_LIMIT:
                if len(self.speech_buffer) > 0:
                    print("\nОбработка команды...")
                    self._process_command()
                self.state = AssistantState.WAITING_FOR_FOLLOWUP
                self.last_activity_time = current_time

    def _handle_followup_mode(self, audio_chunk, current_time):
        """Режим ожидания продолжения"""
        if not self.is_speech(audio_chunk):
            if current_time - self.last_activity_time > self.LONG_SILENCE:
                print("\nВозврат в режим ожидания")
                self._reset_state()
        else:
            self.last_activity_time = current_time

    def _reset_state(self):
        """Сброс состояния"""
        self.state = AssistantState.WAITING_FOR_ACTIVATION
        self.speech_buffer = []
        self.activation_buffer.clear()
        self.last_activity_time = time.time()
        self.on_deactivation()

    def calibrate(self):
        """Быстрая калибровка шума"""
        if self.is_calibrated:
            return

        print("Калибровка микрофона...")
        temp_stream = self.audio_interface.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )

        noise_samples = []
        for _ in range(0, int(self.RATE / self.CHUNK)):  # 1 секунда
            data = temp_stream.read(self.CHUNK)
            noise_samples.append(np.frombuffer(data, dtype=np.int16))

        temp_stream.close()
        self.noise_threshold = np.mean([np.abs(chunk).mean() for chunk in noise_samples]) * 1.8
        self.is_calibrated = True
        print("Калибровка завершена")

    def transcribe_audio(self, audio_data: bytes) -> Optional[str]:
        """Синхронная транскрипция через Deepgram"""
        try:
            options = PrerecordedOptions(
                smart_format=True,
                model="nova-2",
                language="ru"
            )

            response = self.dg_client.listen.prerecorded.v('1').transcribe_file(
                {'buffer': audio_data},
                options
            )

            if (response.results and
                    response.results.channels and
                    response.results.channels[0].alternatives):
                return response.results.channels[0].alternatives[0].transcript

        except Exception as e:
            print(f"Ошибка STT: {e}")
        return None

    def _process_command(self):
        """Обработка записанной команды"""
        if not self.speech_buffer:
            return

        audio_data = self._frames_to_audio_data(self.speech_buffer)
        transcript = self.transcribe_audio(audio_data)

        if transcript:
            self.on_command(transcript)

        self.speech_buffer = []

    def _check_activation(self, audio_data: bytes):
        """Синхронная проверка на команду активации"""
        transcript = self.transcribe_audio(audio_data)
        if transcript and any(word in transcript.lower() for word in ["V", "violet", "слушай, ви."]):
            print("\nАктивация обнаружена!")
            self.state = AssistantState.LISTENING_FOR_COMMAND
            self.speech_buffer = []
            self.last_activity_time = time.time()
            self.on_activation()

    def is_speech(self, audio_chunk) -> bool:
        """Определение речи в аудио чанке"""
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
        filtered_data = lfilter(self.b, self.a, audio_data)
        energy = np.abs(filtered_data).mean()

        spectrum = np.abs(np.fft.rfft(audio_data))
        speech_range = spectrum[int(85 * len(spectrum) / (self.RATE / 2)):
                                int(3000 * len(spectrum) / (self.RATE / 2))]
        spectral_energy = np.sum(speech_range)

        self.speech_window.append(energy)

        if self.energy_threshold is None:
            self.energy_threshold = np.mean(list(self.speech_window)) * 1.0
        else:
            self.energy_threshold = 0.95 * self.energy_threshold + 0.05 * np.mean(list(self.speech_window))

        is_speech = (energy > self.noise_threshold * 1.5 and
                     spectral_energy > self.noise_threshold * 150)

        if is_speech:
            self.speech_count += 1
            self.silence_count = max(0, self.silence_count - 2)
        else:
            self.silence_count += 1
            self.speech_count = max(0, self.speech_count - 1)

        return self.speech_count > 3 and energy > self.energy_threshold * 1.8

    def _frames_to_audio_data(self, frames):
        """Конвертация фреймов в аудио данные"""
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.audio_interface.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(frames))
        return buffer.getvalue()

    def stop(self):
        """Остановка системы"""
        self.is_running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio_interface.terminate()


# Тестовые колбэки
def on_activation():
    print("\nСлушаю...")


def on_command(text: str):
    print(f"\nРаспознано: {text}")


def on_deactivation():
    print("\nОжидаю активации...")


if __name__ == "__main__":
    activator = VoiceActivator(
        on_activation=on_activation,
        on_command=on_command,
        on_deactivation=on_deactivation
    )

    try:
        activator.start()
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        activator.stop()
        print("\nСистема остановлена")