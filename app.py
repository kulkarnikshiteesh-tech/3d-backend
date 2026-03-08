import gradio as gr

def test_app(file, text):
    return f'Thank you! File: {file.name if file else 'none'}, Text: {text}'

demo = gr.Interface(
    fn=test_app,
    inputs=[gr.File(), gr.Textbox()],
    outputs=gr.Textbox()
)

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860)
