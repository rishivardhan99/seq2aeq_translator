import argparse, os, pickle, json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_artifacts(model_dir):
    model = load_model(os.path.join(model_dir,'seq2seq_model'), compile=False)
    with open(os.path.join(model_dir,'eng_tokenizer.pkl'),'rb') as f:
        eng_tok = pickle.load(f)
    with open(os.path.join(model_dir,'fra_tokenizer.pkl'),'rb') as f:
        fra_tok = pickle.load(f)
    with open(os.path.join(model_dir,'meta.json'),'r',encoding='utf-8') as f:
        meta = json.load(f)
    return model, eng_tok, fra_tok, meta

def greedy_decode(model, eng_tok, fra_tok, meta, sentence):
    max_enc = meta.get('max_enc_len',20)
    max_dec = meta.get('max_dec_len',20)
    seq = eng_tok.texts_to_sequences([sentence.lower()])
    enc_seq = pad_sequences(seq, maxlen=max_enc, padding='post')
    # encoder output and states are not directly exposed; here we run model up to encoder by creating sub-models is complex.
    # Simpler approach: use the same model in training mode but feed decoder inputs step by step by using saved model structure is complex.
    # To keep inference simple in this starter repo, we will do a *naive* approach: use the model to predict the whole target sequence
    # by feeding a start token repeated. This works for small demos but for robust production, implement separate encoder/decoder models.
    start_token = '<sos>'
    dec_input_seq = fra_tok.texts_to_sequences([start_token])[0]
    dec_input = pad_sequences([dec_input_seq + [0]*(max_dec-1)], maxlen=max_dec-1, padding='post')
    preds = model.predict([enc_seq, dec_input])
    pred_ids = np.argmax(preds[0], axis=-1)
    # convert ids to words until <eos>
    inv = {v:k for k,v in fra_tok.word_index.items()}
    words = []
    for idx in pred_ids:
        if idx == 0: continue
        w = inv.get(idx, '<unk>')
        if w == '<eos>': break
        if w == '<sos>': continue
        words.append(w)
    return ' '.join(words)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='models')
    parser.add_argument('--sentence', required=True)
    args = parser.parse_args()
    model, eng_tok, fra_tok, meta = load_artifacts(args.model_dir)
    print('Translating:', args.sentence)
    print('=>', greedy_decode(model, eng_tok, fra_tok, meta, args.sentence))
