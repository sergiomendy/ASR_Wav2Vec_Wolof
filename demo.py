from transformers import pipeline
from transformers import Wav2Vec2CTCTokenizer
import gradio as gr

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("tokenizer")

pipe = pipeline("automatic-speech-recognition", model="serge-wilson/wav2vec-base-wolof", tokenizer=tokenizer)  

def transcribe(audio):
    return pipe(audio)["text"]

iface = gr.Interface(
    fn=transcribe, 
    inputs=gr.Audio(source="microphone", type="filepath"), 
    outputs="text",
    title="Wav2Vec Base",
    description="Demo for Wolof speech recognition using a fine-tuned Wav2Vec base model.",
)

iface.launch()
