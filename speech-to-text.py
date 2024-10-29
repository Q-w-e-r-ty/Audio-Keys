# audio_transcription.py

import librosa
import torch
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import IPython.display as display


def load_model_and_processor():
    """Load the pre-trained Wav2Vec2 model and processor."""
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    return processor, model


def load_audio(file_path):
    """Load audio file and check its sampling rate."""
    audio, sampling_rate = librosa.load(file_path, sr=None)
    return audio, sampling_rate


def convert_sampling_rate(audio, original_sr, target_sr=16000):
    """Convert audio to the target sampling rate if necessary."""
    if original_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
    return audio


def transcribe_audio(file_path):
    """Transcribe audio file to text."""
    processor, model = load_model_and_processor()
    
    # Load and process audio
    audio, sampling_rate = load_audio(file_path)
    audio = convert_sampling_rate(audio, sampling_rate)

    # Prepare input values for the model
    input_values = processor(audio, return_tensors='pt', sampling_rate=16000).input_values

    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits

    # Get predicted IDs and decode them to text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcriptions = processor.decode(predicted_ids[0])

    return transcriptions


def give_text_from_audio(location):
    """Main function to get text from audio file at the specified location."""
    return transcribe_audio(location)


# Example of how to use the function
if __name__ == "__main__":
    audio_file_path = "E:/Padhai/SEM/ML/Audio Keys/16000_pcm_speeches/Tanmay_Raw/Tanmay.wav"
    transcription = give_text_from_audio(audio_file_path)
    print("Transcription:", transcription)
