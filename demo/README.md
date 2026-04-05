# Demo

This folder now includes Colab-facing helpers for launching the comparison app.

Quick path in Colab:

```python
!pip install -r requirements-colab.txt
```

```python
from demo.colab_quickstart import print_quickstart, prepare_project_runtime, launch_app
print_quickstart()
prepare_project_runtime(
    mistral_model_id="mistralai/Mistral-7B-Instruct-v0.2",
    secondary_model_id="meta-llama/Llama-3.2-3B-Instruct",
)
launch_app()
```
