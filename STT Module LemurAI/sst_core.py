import os
import wave
import pyaudio
from faster_whisper import WhisperModel  # Assuming this is a custom module

# Constants definition
NEON_GREEN = '\033[32m'
RESET_COLOR = '\033[0m'

# Set environment variable to avoid MKL duplicate library error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def record_chunk(p, stream, file_path, chunk_length=1):
    """
    Records an audio chunk into a file.

    Args:
        p (pyaudio.PyAudio): PyAudio object.
        stream (pyaudio.Stream): PyAudio stream.
        file_path (str): File path where the audio chunk will be saved.
        chunk_length (int): Length of audio chunk in seconds.

    Returns:
        None
    """
    frames = []
    for _ in range(0, int(16000 / 1024 * chunk_length)):
        data = stream.read(1024)
        frames.append(data)

    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()


def transcribe_chunk(model, file_path):
    """
    Transcribes an audio chunk using the Whisper model.

    Args:
        model (WhisperModel): Whisper model object.
        file_path (str): Path to the audio file to transcribe.

    Returns:
        str: Transcription of the audio chunk.
    """
    segments, info = model.transcribe(file_path, beam_size=7)
    transcription = ''.join(segment.text for segment in segments)
    return transcription


def main():
    """
    Main function of the program.
    """
    # Choose the Whisper model
    model_size = "distil-large-v2"
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open audio stream for recording
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

    # Initialize an empty string to accumulate transcriptions
    accumulated_transcription = ""

    try:
        while True:
            # Record an audio chunk
            chunk_file = "temp_chunk.wav"
            record_chunk(p, stream, chunk_file)

            # Transcribe the audio chunk
            transcription = transcribe_chunk(model, chunk_file)
            print(NEON_GREEN + transcription + RESET_COLOR)

            # Remove the temporary file
            os.remove(chunk_file)

            # Add the new transcription to the accumulated transcription
            accumulated_transcription += transcription + " "

    except KeyboardInterrupt:
        print("Stopping...")

        # Write accumulated transcription to a log file
        with open("log.txt", "w") as log_file:
            log_file.write(accumulated_transcription)

    finally:
        # Log the accumulated transcription
        print("LOG", accumulated_transcription)

        # Stop and close the recording stream
        stream.stop_stream()
        stream.close()

        # Terminate PyAudio
        p.terminate()


if __name__ == "__main__":
    main()
