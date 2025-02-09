import pyaudio
import wave
import numpy as np
import time
import subprocess
import os


def record_with_noise_control(output_filename, silence_duration=1.2):
    """
    Профессиональная запись с контролем шума и тишины.
    """
    CHUNK = 1024
    FORMAT = pyaudio.paInt16  # Меняем на Int16 для лучшего качества
    CHANNELS = 1
    RATE = 44100  # Стандартная частота для лучшего качества

    p = pyaudio.PyAudio()

    # Калибровка уровня шума
    print("* Калибрую уровень шума (1 секунда)...")
    calibration_stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    # Собираем данные для калибровки
    noise_samples = []
    for _ in range(0, int(RATE / CHUNK)):
        data = calibration_stream.read(CHUNK)
        noise_samples.append(np.frombuffer(data, dtype=np.int16))

    calibration_stream.stop_stream()
    calibration_stream.close()

    # Вычисляем базовый уровень шума
    noise_threshold = np.mean([np.abs(chunk).mean() for chunk in noise_samples]) * 1.2

    print(f"* Уровень шума откалиброван: {noise_threshold}")
    print("* Начинаю запись... (для остановки - 2 секунды тишины)")

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

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            volume_norm = np.abs(audio_data).mean()

            if volume_norm > noise_threshold:
                frames.append(data)
                is_silent = False
                silence_start = None
                print("▶", end="", flush=True)
            else:
                frames.append(data)
                if not is_silent:
                    silence_start = time.time()
                    is_silent = True
                elif time.time() - silence_start >= silence_duration:
                    print("\n* Обнаружена тишина")
                    break
                print(".", end="", flush=True)

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
    record_with_noise_control(filename)


if __name__ == "__main__":
    main()