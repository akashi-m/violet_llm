import pyaudio
import wave
import numpy as np
import time
import subprocess
import os
from scipy.signal import butter, lfilter
from collections import deque


def butter_bandpass(lowcut, highcut, fs, order=5):
    """Создает полосовой фильтр Баттерворта"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Применяет полосовой фильтр к данным"""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


class VoiceActivityDetector:
    def __init__(self, rate=44100, chunk_size=1024):
        self.rate = rate
        self.chunk_size = chunk_size
        self.speech_window = deque(maxlen=20)  # Увеличено окно истории
        self.energy_threshold = None
        self.speech_count = 0
        self.silence_count = 0

    def is_speech(self, audio_chunk, noise_threshold):
        """Определяет, является ли чанк речью"""
        # Конвертируем байты в numpy массив
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)

        # Применяем полосовой фильтр для выделения частот речи (85-3000 Гц)
        filtered_data = butter_bandpass_filter(audio_data, 85, 3000, self.rate)

        # Вычисляем энергию сигнала
        energy = np.abs(filtered_data).mean()

        # Вычисляем спектральные характеристики
        spectrum = np.abs(np.fft.rfft(audio_data))
        speech_range = spectrum[int(85 * len(spectrum) / (self.rate / 2)):
                                int(3000 * len(spectrum) / (self.rate / 2))]
        spectral_energy = np.sum(speech_range)

        # Добавляем текущую энергию в окно
        self.speech_window.append(energy)

        # Динамически обновляем порог с повышенным коэффициентом
        if self.energy_threshold is None:
            self.energy_threshold = np.mean(list(self.speech_window)) * 1.0  # Увеличен множитель
        else:
            self.energy_threshold = 0.95 * self.energy_threshold + \
                                    0.05 * np.mean(list(self.speech_window))

        # Определяем наличие речи с повышенными порогами
        is_speech = (energy > noise_threshold * 1.5 and  # Увеличен множитель
                     spectral_energy > noise_threshold * 150)  # Увеличен множитель

        if is_speech:
            self.speech_count += 1
            self.silence_count = max(0, self.silence_count - 2)  # Быстрее уменьшаем счетчик тишины
        else:
            self.silence_count += 1
            self.speech_count = max(0, self.speech_count - 1)

        # Требуем больше подтверждений для определения речи
        return (self.speech_count > 3 and  # Увеличено количество подтверждений
                energy > self.energy_threshold * 1.8)  # Увеличен множитель


def record_with_voice_activity_detection(output_filename, silence_duration=1.2):
    """
    Улучшенная запись с определением голосовой активности
    """
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    p = pyaudio.PyAudio()
    vad = VoiceActivityDetector(RATE, CHUNK)

    # Калибровка уровня шума с увеличенным временем
    print("* Калибрую уровень шума (2 секунды)...")
    calibration_stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    noise_samples = []
    for _ in range(0, int(RATE / CHUNK * 2)):  # Увеличено время калибровки
        data = calibration_stream.read(CHUNK)
        noise_samples.append(np.frombuffer(data, dtype=np.int16))

    calibration_stream.stop_stream()
    calibration_stream.close()

    # Увеличиваем множитель порога шума
    noise_threshold = np.mean([np.abs(chunk).mean() for chunk in noise_samples]) * 1.8
    print(f"* Уровень шума откалиброван: {noise_threshold}")
    print("* Начинаю запись... (остановится автоматически после паузы в речи)")

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    frames = []
    silence_start = None
    is_silent = False
    min_speech_duration = 1.0  # Увеличена минимальная длительность речи
    speech_detected = False

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

            if vad.is_speech(data, noise_threshold):
                speech_detected = True
                if is_silent:
                    is_silent = False
                    silence_start = None
                print("▶", end="", flush=True)
            else:
                print(".", end="", flush=True)
                if not is_silent:
                    silence_start = time.time()
                    is_silent = True
                elif time.time() - silence_start >= silence_duration:
                    if speech_detected and len(frames) * CHUNK / RATE > min_speech_duration:
                        print("\n* Обнаружен конец речи")
                        break
                    else:
                        # Сбрасываем счетчик тишины, если запись слишком короткая
                        silence_start = None
                        is_silent = False

    except KeyboardInterrupt:
        print("\n* Остановлено пользователем")

    print("\n* Финализация записи...")

    stream.stop_stream()
    stream.close()
    p.terminate()

    if len(frames) > 0:
        wf = wave.open(output_filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        print("* Воспроизведение...")
        subprocess.run(['afplay', output_filename])
    else:
        print("* Ошибка: Нет данных для сохранения")


def main():
    os.makedirs('recordings', exist_ok=True)
    filename = f"recordings/record.wav"
    record_with_voice_activity_detection(filename)


if __name__ == "__main__":
    main()