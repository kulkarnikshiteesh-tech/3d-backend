import gradio as gr
def dfm(file, surface):
    return f"DFM {surface} File {file.size if file else 0} bytes"
gr.Interface(fn=dfm, inputs=[gr.File(), gr.Dropdown(["Top", "Bottom"])]).launch(server_name="0.0.0.0", server_port=7860)
