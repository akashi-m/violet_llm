import numpy as np
import librosa
import pickle
from collections import deque
from typing import Dict, Optional, List
import matplotlib.pyplot as plt
import pyaudio
import seaborn as sns
from scipy.signal import butter, lfilter


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 5) -> np.ndarray:
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


class VoiceActivityDetector:
    def __init__(self, rate: int = 48000, chunk_size: int = 1024):
        self.rate = rate
        self.chunk_size = chunk_size
        self.speech_window = deque(maxlen=20)
        self.energy_threshold = None
        self.speech_count = 0
        self.silence_count = 0
        self.calibrated = False
        self.noise_threshold = None

        # Инициализация PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self._init_audio_stream()

    def _init_audio_stream(self):
        """Инициализация аудио потока"""
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            print("[VAD] Аудио поток инициализирован")
        except Exception as e:
            print(f"[ОШИБКА] Не удалось инициализировать аудио поток: {e}")
            self.stream = None

    def get_audio_frame(self) -> Optional[np.ndarray]:
        """Получение фрейма аудио с микрофона"""
        if not self.stream:
            self._init_audio_stream()
            if not self.stream:
                return None

        try:
            # Чтение данных
            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            # Преобразование в numpy array
            audio_frame = np.frombuffer(data, dtype=np.int16)

            # Калибровка при первом запуске
            if not self.calibrated:
                self.calibrate()

            return audio_frame

        except Exception as e:
            print(f"[ОШИБКА] Ошибка получения аудио фрейма: {e}")
            # Попытка переинициализации потока при ошибке
            self.stream = None
            return None

    def __del__(self):
        """Освобождение ресурсов при удалении объекта"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()

    def calibrate(self, duration: float = 2.0) -> float:
        """Калибровка порога шума"""
        print("\n[VAD] Калибровка уровня шума...")
        samples = []
        required_chunks = int((self.rate * duration) / self.chunk_size)

        for _ in range(required_chunks):
            try:
                # Здесь будет получение аудио от микрофона
                audio_chunk = np.random.randn(self.chunk_size) * 0.01  # Заглушка
                samples.append(np.abs(audio_chunk).mean())
            except Exception as e:
                print(f"[VAD] Ошибка калибровки: {e}")
                continue

        if samples:
            self.noise_threshold = np.mean(samples) * 1.8
            self.calibrated = True
            print(f"[VAD] Порог шума установлен: {self.noise_threshold:.6f}")
            return self.noise_threshold
        return 0.0

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        if not self.calibrated:
            self.calibrate()

        try:
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

            is_speech = (energy > self.noise_threshold * 1.5 and
                         spectral_energy > self.noise_threshold * 150)

            if is_speech:
                self.speech_count += 1
                self.silence_count = max(0, self.silence_count - 2)
            else:
                self.silence_count += 1
                self.speech_count = max(0, self.speech_count - 1)

            return (self.speech_count > 3 and energy > self.energy_threshold * 1.8)

        except Exception as e:
            print(f"[VAD] Ошибка обработки аудио: {e}")
            return False


class VoiceAnalyzer:
    def __init__(self, sample_rate: int = 48000, confidence_threshold: float = 0.85):
        self.sample_rate = sample_rate
        self.confidence_threshold = confidence_threshold
        self.voice_profiles = []
        self.metrics = {
            'total_comparisons': 0,
            'positive_matches': 0,
            'avg_similarity': 0.0
        }

    def extract_features(self, audio_data: np.ndarray) -> Optional[Dict]:
        try:
            # Нормализация аудио
            y = librosa.util.normalize(audio_data)

            # Извлечение характеристик
            mfcc = librosa.feature.mfcc(y=y, sr=self.sample_rate, n_mfcc=13)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=self.sample_rate)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sample_rate)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)

            features = {
                'mfcc_mean': np.mean(mfcc, axis=1),
                'mfcc_std': np.std(mfcc, axis=1),
                'centroid_mean': np.mean(spectral_centroid),
                'rolloff_mean': np.mean(spectral_rolloff),
                'zcr_mean': np.mean(zero_crossing_rate)
            }

            return features

        except Exception as e:
            print(f"[VA] Ошибка извлечения характеристик: {e}")
            return None

    def create_voice_profile(self, audio_files: List[str]) -> bool:
        try:
            profiles = []
            for file in audio_files:
                # Загрузка аудио
                y, _ = librosa.load(file, sr=self.sample_rate)
                features = self.extract_features(y)
                if features:
                    profiles.append(features)

            if profiles:
                self.voice_profiles = profiles
                return True
            return False

        except Exception as e:
            print(f"[VA] Ошибка создания профиля: {e}")
            return False

    def compare_voice(self, audio_data: np.ndarray) -> float:
        try:
            # Конвертация в float32 и нормализация
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max

            features = self.extract_features(audio_data)
            if not features or not self.voice_profiles:
                return 0.0

            similarities = []
            for profile in self.voice_profiles:
                similarity = self._calculate_similarity(features, profile)
                similarities.append(similarity)

            max_similarity = max(similarities)

            # Обновление метрик
            self.metrics['total_comparisons'] += 1
            if max_similarity >= self.confidence_threshold:
                self.metrics['positive_matches'] += 1
            self.metrics['avg_similarity'] = (
                    (self.metrics['avg_similarity'] * (self.metrics['total_comparisons'] - 1) +
                     max_similarity) / self.metrics['total_comparisons']
            )

            return max_similarity

        except Exception as e:
            print(f"[VA] Ошибка сравнения голоса: {e}")
            return 0.0

    def _calculate_similarity(self, features1: Dict, features2: Dict) -> float:
        try:
            # Косинусное сходство для MFCC
            mfcc_sim = np.dot(features1['mfcc_mean'], features2['mfcc_mean']) / (
                    np.linalg.norm(features1['mfcc_mean']) * np.linalg.norm(features2['mfcc_mean']))

            # Сходство других характеристик
            centroid_diff = abs(features1['centroid_mean'] - features2['centroid_mean'])
            rolloff_diff = abs(features1['rolloff_mean'] - features2['rolloff_mean'])

            # Взвешенная комбинация метрик
            similarity = (0.6 * mfcc_sim +
                          0.2 * (1 - centroid_diff / features2['centroid_mean']) +
                          0.2 * (1 - rolloff_diff / features2['rolloff_mean']))

            return max(0.0, min(1.0, similarity))

        except Exception as e:
            print(f"[VA] Ошибка вычисления схожести: {e}")
            return 0.0

    def save_profile(self, path: str) -> bool:
        try:
            with open(path, 'wb') as f:
                pickle.dump(self.voice_profiles, f)
            print(f"[VA] Профиль сохранен: {path}")
            return True
        except Exception as e:
            print(f"[VA] Ошибка сохранения профиля: {e}")
            return False

    def load_profile(self, path: str) -> bool:
        try:
            with open(path, 'rb') as f:
                self.voice_profiles = pickle.load(f)
            print(f"[VA] Профиль загружен: {path}")
            return True
        except Exception as e:
            print(f"[VA] Ошибка загрузки профиля: {e}")
            return False

    def visualize_profile(self, save_path: Optional[str] = None) -> None:
        if not self.voice_profiles:
            print("[VA] Нет загруженного профиля для визуализации")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Характеристики голосового профиля", fontsize=16)

        # MFCC тепловая карта
        mfcc_means = np.array([p['mfcc_mean'] for p in self.voice_profiles])
        sns.heatmap(mfcc_means, ax=axes[0, 0], cmap='viridis')
        axes[0, 0].set_title('MFCC Profile')
        axes[0, 0].set_xlabel('MFCC Coefficients')
        axes[0, 0].set_ylabel('Sample')

        # Спектральный центроид
        centroids = [p['centroid_mean'] for p in self.voice_profiles]
        axes[0, 1].plot(centroids, 'ro-')
        axes[0, 1].set_title('Spectral Centroid')
        axes[0, 1].set_xlabel('Sample')
        axes[0, 1].set_ylabel('Frequency (Hz)')

        # Распределение MFCC
        mfcc_flat = mfcc_means.flatten()
        sns.histplot(mfcc_flat, ax=axes[1, 0], kde=True)
        axes[1, 0].set_title('MFCC Distribution')
        axes[1, 0].set_xlabel('MFCC Value')

        # Метрики сравнения
        if self.metrics['total_comparisons'] > 0:
            match_rate = self.metrics['positive_matches'] / self.metrics['total_comparisons']
            axes[1, 1].bar(['Match Rate', 'Avg Similarity'],
                           [match_rate, self.metrics['avg_similarity']])
            axes[1, 1].set_title('Voice Analysis Metrics')
            axes[1, 1].set_ylim(0, 1)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"[VA] Визуализация сохранена: {save_path}")
        else:
            plt.show()