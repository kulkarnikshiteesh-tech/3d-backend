import gradio as gr
def dfm(file, surface):
return f"DFM for {surface}\\nFile size: {file.size if file else 0} bytes"
demo = gr.Interface(
fn=dfm, inputs=[gr.File(), gr.Dropdown(["Top", "Bottom"])], outputs=gr.Text()
 )
demo.launch(server_name="0.0.0.0", server_port=7860)
