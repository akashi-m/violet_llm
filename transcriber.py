import pyaudio
import numpy as np
import time
import os
from scipy.signal import butter, lfilter
from collections import deque
from deepgram import DeepgramClient, PrerecordedOptions
from dotenv import load_dotenv
import io
import wave

load_dotenv()


class VoiceActivityDetector:
    def __init__(self, rate=44100, chunk_size=1024):
        self.rate = rate
        self.chunk_size = chunk_size
        self.speech_window = deque(maxlen=20)
        self.energy_threshold = None
        self.speech_count = 0
        self.silence_count = 0

    def is_speech(self, audio_chunk, noise_threshold):
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
        filtered_data = butter_bandpass_filter(audio_data, 85, 3000, self.rate)
        energy = np.abs(filtered_data).mean()

        spectrum = np.abs(np.fft.rfft(audio_data))
        speech_range = spectrum[int(85 * len(spectrum) / (self.rate / 2)):
                                int(3000 * len(spectrum) / (self.rate / 2))]
        spectral_energy = np.sum(speech_range)

        self.speech_window.append(energy)

        if self.energy_threshold is None:
            self.energy_threshold = np.mean(list(self.speech_window)) * 1.0
        else:
            self.energy_threshold = 0.95 * self.energy_threshold + 0.05 * np.mean(list(self.speech_window))

        is_speech = (energy > noise_threshold * 1.5 and spectral_energy > noise_threshold * 150)

        if is_speech:
            self.speech_count += 1
            self.silence_count = max(0, self.silence_count - 2)
        else:
            self.silence_count += 1
            self.speech_count = max(0, self.speech_count - 1)

        return (self.speech_count > 3 and energy > self.energy_threshold * 1.8)


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def frames_to_audio_data(frames, channels=1, rate=44100, sample_width=2):
    """Конвертирует фреймы в аудио буфер"""
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
    return buffer.getvalue()


def record_and_transcribe():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    dg_client = DeepgramClient(os.getenv('DEEPGRAM_API_KEY'))
    p = pyaudio.PyAudio()
    vad = VoiceActivityDetector(RATE, CHUNK)

    print("* Калибрую уровень шума (2 секунды)...")
    calibration_stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    noise_samples = []
    for _ in range(0, int(RATE / CHUNK * 2)):
        data = calibration_stream.read(CHUNK)
        noise_samples.append(np.frombuffer(data, dtype=np.int16))

    calibration_stream.stop_stream()
    calibration_stream.close()

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
    min_speech_duration = 1.0
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
                elif time.time() - silence_start >= 1.2:
                    if speech_detected and len(frames) * CHUNK / RATE > min_speech_duration:
                        print("\n* Обнаружен конец речи")
                        break
                    else:
                        silence_start = None
                        is_silent = False

    except KeyboardInterrupt:
        print("\n* Остановлено пользователем")

    print("\n* Обработка записи...")

    stream.stop_stream()
    stream.close()
    p.terminate()

    if len(frames) > 0:
        audio_data = frames_to_audio_data(frames, CHANNELS, RATE, p.get_sample_size(FORMAT))

        try:
            options = PrerecordedOptions(
                smart_format=True,
                model="nova-2",
                language="ru"
            )

            response = dg_client.listen.prerecorded.v('1').transcribe_file(
                {'buffer': audio_data}, options
            )

            if (response.results and
                    response.results.channels and
                    response.results.channels[0].alternatives):
                transcript = response.results.channels[0].alternatives[0].transcript
                confidence = response.results.channels[0].alternatives[0].confidence
                print(f"\nУверенность: {confidence:.2f}")
                print(f"Текст: {transcript}")
                return transcript

        except Exception as e:
            print(f"Ошибка транскрипции: {str(e)}")

    return None


if __name__ == "__main__":
    record_and_transcribe()