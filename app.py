import os
os.system("pip install -r requirements.txt")
import gradio as gr
import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize

def process_audio(audio_file, do_noise_reduction, do_normalization, do_eq, do_pitch_shift):
    if audio_file is None:
        return None

    y, sr = librosa.load(audio_file, sr=None)

    if do_noise_reduction:
        y = nr.reduce_noise(y=y, sr=sr)

    sf.write("temp.wav", y, sr)

    audio = AudioSegment.from_wav("temp.wav")

    if do_normalization:
        audio = normalize(audio)

    if do_eq:
        audio = audio.high_pass_filter(120)
        audio = audio.low_pass_filter(7500)

    audio.export("processed.wav", format="wav")

    if do_pitch_shift:
        y, sr = librosa.load("processed.wav", sr=None)
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
        sf.write("processed.wav", y, sr)

    return "processed.wav"

demo = gr.Interface(
    fn=process_audio,
    inputs=[
        gr.Audio(type="filepath", label="Upload Audio"),
        gr.Checkbox(label="Noise Reduction", value=True),
        gr.Checkbox(label="Normalize Volume", value=True),
        gr.Checkbox(label="EQ for Clarity", value=True),
        gr.Checkbox(label="Pitch Shift (Voice Change)", value=False),
    ],
    outputs=gr.Audio(type="filepath", label="Download Enhanced Audio"),
    title="Voice Enhancer Tool",
    description="Upload your voice, apply enhancements like noise reduction, EQ, voice change and download it!"
)

demo.launch()
