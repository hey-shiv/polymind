from config import is_colab_runtime
from ui.gradio_app import demo


if __name__ == "__main__":
    demo.launch(share=is_colab_runtime())
