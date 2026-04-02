import gradio as gr
import torch
from pathlib import Path

from generate import load_model, generate_text
from src.device import get_default_device

MODEL_PATH = Path("model.pt")

device = get_default_device()

model, tokenizer = load_model(MODEL_PATH, device)


def generate(prompt, temperature, max_tokens):
    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        device=device,
    )
    return output


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(lines=3, placeholder="Once upon a time..."),
        gr.Slider(0, 1.5, value=1.0, label="Temperature"),
        gr.Slider(10, 300, value=100, step=10, label="Max New Tokens"),
    ],
    outputs="text",
    title="PolyMind",
    description="Local LLM playground built on a compact TinyStories transformer",
)

demo.launch()
