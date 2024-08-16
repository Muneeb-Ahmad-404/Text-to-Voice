import streamlit as st
import torch as torch
from transformers import AutoProcessor, AutoModelForTextToSpectrogram, pipeline
import soundfile as sf
from datasets import load_dataset

# Load models and processor
processor = AutoProcessor.from_pretrained("microsoft/speecht5_tts")
model = AutoModelForTextToSpectrogram.from_pretrained("microsoft/speecht5_tts")
synthesiser = pipeline("text-to-speech", model="microsoft/speecht5_tts")
            
# Load speaker embeddings dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

def get_speaker_embedding(index):
    return torch.tensor(embeddings_dataset[index]["xvector"]).unsqueeze(0)

# Streamlit app
st.title("Text-to-Speech with SpeechT5")

text_input = st.text_area("Enter text to convert to speech:", "Hello, my dog is cooler than you! Yeah, are you angry? Huh Cry about it!")

# Optionally select a speaker embedding
speaker_index = st.slider("Select speaker index (0-10000):", 0, 10000, 7306)
speaker_embedding = get_speaker_embedding(speaker_index)

if st.button("Generate Speech"):
    with st.spinner("Generating speech..."):
        # Synthesize speech
        speech = synthesiser(text_input, forward_params={"speaker_embeddings": speaker_embedding})

        # Save to file
        audio_path = "speech.wav"
        sf.write(audio_path, speech["audio"], samplerate=speech["sampling_rate"])

        # Display results
        st.audio(audio_path, format="audio/wav")
        st.success("Speech generated successfully!")

# Add a download button for the generated audio file
st.download_button(label="Download Speech", data=open("speech.wav", "rb").read(), file_name="speech.wav", mime="audio/wav")

st.title("Speech")