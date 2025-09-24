# Seq2Seq Language Translator (English -> French)
This repository contains a minimal, ready-to-run implementation for a Seq2Seq translator using Keras (TensorFlow). It includes training and inference scripts, a tiny sample dataset (English-French), and instructions to reproduce results. Designed for easy submission to code challenges or GitHub.

## Contents
- `data/sample_eng_fra.txt` — small parallel corpus (tab-separated: english \t french).
- `train.py` — training script that builds an encoder-decoder LSTM model with attention and saves the model and tokenizers.
- `infer.py` — inference script to load saved artifacts and translate input English sentences to French (greedy decoding).
- `demo.py` — demo runner that quickly trains for a couple epochs on the tiny dataset and demonstrates translation.
- `requirements.txt` — Python dependencies.
- `submission_checklist.txt` — quick checklist of files to submit.
- `models/` — folder where trained model and tokenizer files are saved after running training.
- `.gitignore`, `LICENSE`

## How to run (example)
1. Create virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate   # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```
2. Quick demo (small data, quick):
```bash
python demo.py
```
This will train for a few epochs on the tiny sample and show sample translations.

3. To train properly on a larger dataset, replace `data/sample_eng_fra.txt` with your full parallel corpus (one sentence pair per line, tab-separated), then run:
```bash
python train.py --data_path data/sample_eng_fra.txt --epochs 30 --batch_size 64
```

4. After training, run inference:
```bash
python infer.py --model_dir models --sentence "I love you"
```

## Deliverables (for submission)
- Code notebook / scripts: `train.py`, `infer.py`, `demo.py`
- Model + tokenizers: saved under `models/` after running training
- Translation demo: `demo.py` shows how to run a small demo

## Notes
- This is a small, readable starter. For production, use larger datasets, subword tokenization (SentencePiece/BPE), teacher forcing, hyperparameter tuning, and better attention / beam search for inference.
- The provided sample is intentionally small to make the demo fast and runnable on modest machines.
