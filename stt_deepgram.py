import os
import wave
from deepgram import DeepgramClient, PrerecordedOptions
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def check_wav_file(file_path):
    """
    Проверяет и выводит информацию о WAV файле
    """
    try:
        with wave.open(file_path, 'rb') as wav:
            print("\nWAV File Info:")
            print(f"Number of channels: {wav.getnchannels()}")
            print(f"Sample width: {wav.getsampwidth()} bytes")
            print(f"Frame rate (sampling frequency): {wav.getframerate()} Hz")
            print(f"Number of frames: {wav.getnframes()}")
            print(f"Duration: {wav.getnframes() / wav.getframerate():.2f} seconds")

            # Проверяем, что файл не пустой
            if wav.getnframes() == 0:
                print("WARNING: File contains no frames!")
                return False

            return True
    except Exception as e:
        print(f"Error checking WAV file: {e}")
        return False


def transcribe_wav_file(file_path):
    """
    Transcribes a WAV file using Deepgram's API.
    """
    # Проверяем файл перед отправкой
    if not check_wav_file(file_path):
        return None

    try:
        # Инициализируем клиент
        dg_client = DeepgramClient(os.getenv('DEEPGRAM_API_KEY'))

        with open(file_path, 'rb') as audio:
            # Настраиваем опции для транскрипции
            options = PrerecordedOptions(
                smart_format=True,
                model="nova-2",
               # tier="enhanced",
                language="ru"
            )

            # Отправляем аудио на транскрипцию
            response = dg_client.listen.prerecorded.v('1').transcribe_file(
                {'buffer': audio}, options
            )

            # Выводим метаданные для диагностики
            print("\nMetadata:")
            print(f"Request ID: {response.metadata.request_id}")
            print(f"Duration: {response.metadata.duration}")
            print(f"Channels: {response.metadata.channels}")

            # Проверяем наличие транскрипции
            if (response.results and
                    response.results.channels and
                    response.results.channels[0].alternatives):

                transcript = response.results.channels[0].alternatives[0].transcript
                confidence = response.results.channels[0].alternatives[0].confidence
                print(f"\nConfidence: {confidence}")
                return transcript
            else:
                print("No transcription found in response")
                return None

    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        return None


if __name__ == "__main__":
    file_path = 'recordings/record.wav'

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found!")
    else:
        transcript = transcribe_wav_file(file_path)
        if transcript:
            print("\nTranscription:", transcript)
        else:
            print("\nFailed to transcribe the audio.")