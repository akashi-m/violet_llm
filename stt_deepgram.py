import os
from deepgram import DeepgramClient, PrerecordedOptions
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the Deepgram client
dg_client = DeepgramClient(os.getenv('DEEPGRAM_API_KEY'))

def transcribe_wav_file(file_path):
    """
    Transcribes a WAV file using Deepgram's API.

    Args:
        file_path (str): Path to the WAV file.

    Returns:
        str: Transcribed text from the WAV file.
    """
    try:
        # Open the audio file
        with open(file_path, 'rb') as audio:
            # Set options for transcription
            options = PrerecordedOptions(
                smart_format=True,  # Enable smart formatting for better readability
                model="nova-2",       # Use Deepgram's latest model
            )

            # Send the audio file to Deepgram for transcription
            response = dg_client.listen.prerecorded.v('1').transcribe_file(
                {'buffer': audio}, options
            )
            print(response)
            # Extract the transcript from the response
            transcript = response['results']['channels'][0]['alternatives'][0]['transcript']
            return transcript

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage
if __name__ == "__main__":
    file_path = 'recordings/record.wav'
    transcript = transcribe_wav_file(file_path)
    if transcript:
        print("Transcription:", transcript)
    else:
        print("Failed to transcribe the audio.")