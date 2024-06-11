import torch
import torchaudio
import librosa
from voice_encoder.py import VoiceEncoder
from text_translator.py import TextTranslator
from speech_synthesizer.py import SpeechSynthesizer

def load_audio(file_path):
    waveform, sr = librosa.load(file_path, sr=16000)
    return waveform, sr

def save_audio(file_path, waveform, sr):
    torchaudio.save(file_path, waveform, sr)

# Initialize models
voice_encoder = VoiceEncoder()
text_translator = TextTranslator(src_lang='en', tgt_lang='es')
speech_synthesizer = SpeechSynthesizer()

# Load pre-trained model weights if available
# voice_encoder.load_state_dict(torch.load('voice_encoder.pth'))
# speech_synthesizer.load_state_dict(torch.load('speech_synthesizer.pth'))

# Load input speech
input_speech_path = 'input_speech.wav'
waveform, sr = load_audio(input_speech_path)
waveform_tensor = torch.tensor(waveform).unsqueeze(0)

# Encode voice characteristics
encoded_voice = voice_encoder(waveform_tensor)

# Translate text
text = "Hello, how are you?"
translated_text = text_translator.translate(text)[0]
print(f"Translated text: {translated_text}")

# Synthesize speech
synthesized_speech = speech_synthesizer(encoded_voice)

# Save the output speech
output_speech_path = 'output_speech.wav'
save_audio(output_speech_path, synthesized_speech.detach().cpu(), sr)

print(f"Synthesized speech saved to: {output_speech_path}")
