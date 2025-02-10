import torch
import soundfile as sf
import subprocess
import tempfile
import os
import ssl
from omegaconf import OmegaConf

class TTSPlayer:
    def __init__(self, speaker="kseniya", sample_rate=48000, put_accent=True, put_yo=True):
        # Отключаем проверку SSL (фиксим ошибку)
        ssl._create_default_https_context = ssl._create_unverified_context

        # Скачиваем файл модели
        torch.hub.download_url_to_file('https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml',
                                       'latest_silero_models.yml',
                                       progress=False)

        # Загружаем параметры модели
        models = OmegaConf.load('latest_silero_models.yml')

        # Определяем устройство (CUDA или CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Загружаем Silero TTS
        self.model, _ = torch.hub.load(repo_or_dir="snakers4/silero-models",
                                       model="silero_tts",
                                       language="ru",
                                       speaker="v3_1_ru")
        self.model.to(self.device)

        # Параметры
        self.speaker = speaker
        self.sample_rate = sample_rate
        self.put_accent = put_accent
        self.put_yo = put_yo

    def say(self, text):
        """Озвучивает текст и воспроизводит его."""
        audio = self.model.apply_tts(text=text, speaker=self.speaker,
                                     sample_rate=self.sample_rate,
                                     put_accent=self.put_accent, put_yo=self.put_yo)

        # Создаем временный WAV-файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            filename = tmpfile.name
            sf.write(filename, audio, self.sample_rate)

        # Воспроизводим аудио
        if os.name == "posix":
            subprocess.run(f"afplay {filename}", shell=True)  # macOS
            # subprocess.run(f"aplay {filename}", shell=True)  # Linux
        else:
            subprocess.run(f"start {filename}", shell=True)  # Windows

        # Удаляем временный файл
        os.remove(filename)

# 🔥 Использование
tts = TTSPlayer()
tts.say("Теперь звук должен воспроизводиться нормально! Без искажений и на комфортной громкости.")
