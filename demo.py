# Quick demo: train for very few epochs on tiny data and run inference
import os
from train import main as train_main
import subprocess, sys, pickle, json, time
# train quickly
os.makedirs('models', exist_ok=True)
train_args = ['--data_path', 'data/sample_eng_fra.txt', '--epochs', '6', '--batch_size', '8', '--model_dir', 'models']
sys.argv = ['train.py'] + train_args
print('Starting quick demo training... (this is tiny data, short epochs)')
train_main(argparse.Namespace(data_path='data/sample_eng_fra.txt', model_dir='models', num_words=5000, max_enc_len=20, max_dec_len=20, embed_dim=64, latent_dim=128, batch_size=8, epochs=6, num_examples=None))
# simple inference run
from infer import load_artifacts, greedy_decode
model, eng_tok, fra_tok, meta = load_artifacts('models')
sentences = ['hello', 'how are you?', 'i love you', 'what is your name?']
for s in sentences:
    print(s, '->', greedy_decode(model, eng_tok, fra_tok, meta, s))
