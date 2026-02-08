import gradio as gr
pip install transformers
pip install torch

def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)


demo = gr.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs=["text"],
)

demo.launch(server_name="127.0.0.1", server_port=7860)
