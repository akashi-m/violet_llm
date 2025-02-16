from pydub import AudioSegment
import os


def convert_m4a_to_wav(input_path: str, output_path: str = None) -> str:
    """
    Конвертирует .m4a файл в .wav с нужными параметрами

    Args:
        input_path: путь к входному .m4a файлу
        output_path: путь для сохранения .wav файла (опционально)

    Returns:
        str: путь к сконвертированному файлу
    """
    try:
        # Если выходной путь не указан, создаем его из входного
        if not output_path:
            output_path = os.path.splitext(input_path)[0] + '.wav'

        # Загружаем аудио
        audio = AudioSegment.from_file(input_path, format="m4a")

        # Установка параметров
        audio = audio.set_frame_rate(48000)  # 48kHz
        audio = audio.set_channels(1)  # Mono
        audio = audio.set_sample_width(2)  # 16 bit

        # Сохраняем в WAV
        audio.export(output_path, format="wav")

        print(f"Конвертация завершена. Файл сохранен: {output_path}")
        return output_path

    except Exception as e:
        print(f"Ошибка при конвертации: {str(e)}")
        return None

# Пример использования:
converted_file = convert_m4a_to_wav("data/activation_voices_raw/Эй Ви .m4a", "data/activation_voices/v8.wav")
print(converted_file)