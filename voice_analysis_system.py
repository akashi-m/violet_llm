import numpy as np
import librosa
import pickle
from typing import Dict, List, Tuple
import os

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


class VoiceAnalyzer:
    def __init__(self, sample_rate=48000, confidence_threshold=0.85):
        self.confidence_threshold = confidence_threshold
        self.sample_rate = sample_rate
        self.voice_profiles = []

    def extract_features(self, audio_path: str) -> Dict:
        """Извлекает характеристики голоса из аудиофайла"""
        try:
            # Загрузка аудио
            y, sr = librosa.load(audio_path, sr=self.sample_rate)

            # Основные характеристики
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

            # Усреднение характеристик
            features = {
                'mfcc_mean': np.mean(mfcc, axis=1),
                'mfcc_std': np.std(mfcc, axis=1),
                'centroid_mean': np.mean(spectral_centroid),
                'rolloff_mean': np.mean(spectral_rolloff)
            }

            return features

        except Exception as e:
            print(f"Ошибка при извлечении характеристик: {str(e)}")
            return None

    def create_voice_profile(self, audio_files: List[str]) -> bool:
        """Создает профиль голоса на основе нескольких записей"""
        try:
            profiles = []
            for file in audio_files:
                features = self.extract_features(file)
                if features:
                    profiles.append(features)

            if profiles:
                self.voice_profiles = profiles
                return True
            return False

        except Exception as e:
            print(f"Ошибка при создании профиля: {str(e)}")
            return False

    def save_profile(self, path: str) -> bool:
        """Сохраняет профиль голоса"""
        try:
            with open(path, 'wb') as f:
                pickle.dump(self.voice_profiles, f)
            print(f"Профиль сохранен: {path}")
            return True
        except Exception as e:
            print(f"Ошибка при сохранении профиля: {str(e)}")
            return False

    def load_profile(self, path: str) -> bool:
        """Загружает профиль голоса"""
        try:
            with open(path, 'rb') as f:
                self.voice_profiles = pickle.load(f)
            return True
        except Exception as e:
            print(f"Ошибка при загрузке профиля: {str(e)}")
            return False

    def compare_voice(self, audio_data: np.ndarray) -> float:
        """Сравнивает входящий аудиосигнал с профилем голоса"""
        try:
            # Извлекаем характеристики из входящего аудио
            features = self.extract_features(audio_data)
            if not features or not self.voice_profiles:
                return 0.0
        except Exception as e:
                print(f"Ошибка при вычислении схожести: {str(e)}")
                return 0.0

    def visualize_profile(self, save_path: Optional[str] = None) -> None:
        """Визуализирует характеристики голосового профиля"""
        if not self.voice_profiles:
            print("Нет загруженного профиля для визуализации")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Характеристики голосового профиля", fontsize=16)

        # MFCC средние значения
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

        # Порог уверенности
        axes[1, 1].axhline(y=self.confidence_threshold, color='r', linestyle='--',
                           label=f'Threshold ({self.confidence_threshold})')
        axes[1, 1].set_title('Confidence Threshold')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Визуализация сохранена: {save_path}")
        else:
            plt.show()

            # Вычисляем схожесть с каждым профилем
            similarities = []
            for profile in self.voice_profiles:
                similarity = self._calculate_similarity(features, profile)
                similarities.append(similarity)

            # Возвращаем максимальную схожесть
            return max(similarities)




def _calculate_similarity(self, features1: Dict, features2: Dict) -> float:
    """Вычисляет схожесть между двумя наборами характеристик"""
    try:
        # Косинусное сходство для MFCC
        mfcc_sim = np.dot(features1['mfcc_mean'], features2['mfcc_mean']) / (
                np.linalg.norm(features1['mfcc_mean']) * np.linalg.norm(features2['mfcc_mean']))

        # Сходство центроидов и rolloff
        centroid_diff = abs(features1['centroid_mean'] - features2['centroid_mean'])
        rolloff_diff = abs(features1['rolloff_mean'] - features2['rolloff_mean'])

        # Взвешенная комбинация метрик
        similarity = 0.6 * mfcc_sim + 0.2 * (1 - centroid_diff) + 0.2 * (1 - rolloff_diff)

        return max(0.0, min(1.0, similarity))

    except Exception as e:
        print(f"Ошибка при вычислении схожести: {str(e)}")
        return 0.0

analyzer = VoiceAnalyzer(confidence_threshold=0.85)
files = [f"data/activation_voices/v{i}.wav" for i in range(1, 9)]
analyzer.create_voice_profile(files)

analyzer.save_profile("voice_profile.pkl")
analyzer.visualize_profile("voice_profile_analysis.png")