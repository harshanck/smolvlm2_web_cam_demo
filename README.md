# Python: SmolVLM2 Webcam Demo

This is a simple example demonstrating how to perform real-time local visual inference using [SmolVLM2](https://huggingface.co/blog/smolvlm2) models. The model reads webcam frames and provides short descriptions.

### Supported Models

- [SmolVLM2-256M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct)
- [SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct)
- [SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct)

### Requirements

- Python 3.12.4
- torch==2.5.1+cu124
- torchaudio==2.5.1+cu124
- torchvision==0.20.1+cu124
- transformers
- opencv-python
- Pillow

### Installation

1. Clone this repository
```bash
git clone <repo_url>
cd <repo_folder>
```

2. Install the requirements
```bash
pip install -r requirements.txt
```

3. Run the demo
```bash
python main.py
```
