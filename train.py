import argparse
import os
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.layers import Attention
from tensorflow.keras.optimizers import Adam

def load_data(path, num_examples=None):
    pairs = []
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if num_examples and i>=num_examples: break
            line = line.strip()
            if not line: continue
            parts = line.split('\t')
            if len(parts) != 2: continue
            pairs.append((parts[0].lower(), parts[1].lower()))
    return pairs

def fit_tokenizers(pairs, num_words=10000):
    eng_texts = [p[0] for p in pairs]
    fra_texts = ['<sos> ' + p[1] + ' <eos>' for p in pairs]
    eng_tokenizer = Tokenizer(num_words=num_words, oov_token='<oov>')
    fra_tokenizer = Tokenizer(num_words=num_words, oov_token='<oov>')
    eng_tokenizer.fit_on_texts(eng_texts)
    fra_tokenizer.fit_on_texts(fra_texts)
    return eng_tokenizer, fra_tokenizer

def sequences_from_tokenizers(pairs, eng_tok, fra_tok, max_enc_len=20, max_dec_len=20):
    eng_texts = [p[0] for p in pairs]
    fra_texts = ['<sos> ' + p[1] + ' <eos>' for p in pairs]
    enc_seq = eng_tok.texts_to_sequences(eng_texts)
    dec_seq = fra_tok.texts_to_sequences(fra_texts)
    enc_seq = pad_sequences(enc_seq, maxlen=max_enc_len, padding='post')
    dec_seq = pad_sequences(dec_seq, maxlen=max_dec_len, padding='post')
    decoder_input = dec_seq[:, :-1]
    decoder_target = dec_seq[:, 1:]
    return enc_seq, decoder_input, decoder_target

def build_model(enc_vocab, dec_vocab, enc_timesteps=20, dec_timesteps=20, embed_dim=128, latent_dim=256):
    # Encoder
    encoder_inputs = Input(shape=(enc_timesteps,), name='encoder_inputs')
    enc_emb = Embedding(enc_vocab, embed_dim, mask_zero=True)(encoder_inputs)
    encoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='encoder_lstm')
    enc_outs, state_h, state_c = encoder_lstm(enc_emb)
    # Decoder
    decoder_inputs = Input(shape=(dec_timesteps-1,), name='decoder_inputs')
    dec_emb = Embedding(dec_vocab, embed_dim, mask_zero=True)(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
    dec_outs, _, _ = decoder_lstm(dec_emb, initial_state=[state_h, state_c])
    # Attention
    attention = Attention(name='attention_layer')([dec_outs, enc_outs])
    concat = Concatenate(axis=-1)([dec_outs, attention])
    outputs = TimeDistributed(Dense(dec_vocab, activation='softmax'))(concat)
    model = Model([encoder_inputs, decoder_inputs], outputs)
    model.compile(optimizer=Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main(args):
    pairs = load_data(args.data_path, num_examples=args.num_examples)
    print(f'Loaded {len(pairs)} sentence pairs.')
    eng_tok, fra_tok = fit_tokenizers(pairs, num_words=args.num_words)
    enc_seq, dec_in, dec_target = sequences_from_tokenizers(pairs, eng_tok, fra_tok, max_enc_len=args.max_enc_len, max_dec_len=args.max_dec_len)
    dec_target = np.expand_dims(dec_target, -1)  # for sparse_categorical_crossentropy with TimeDistributed
    model = build_model(enc_vocab=min(args.num_words, len(eng_tok.word_index)+1),
                        dec_vocab=min(args.num_words, len(fra_tok.word_index)+1),
                        enc_timesteps=args.max_enc_len,
                        dec_timesteps=args.max_dec_len,
                        embed_dim=args.embed_dim,
                        latent_dim=args.latent_dim)
    model.summary()
    model.fit([enc_seq, dec_in], dec_target, batch_size=args.batch_size, epochs=args.epochs, validation_split=0.1)
    os.makedirs(args.model_dir, exist_ok=True)
    model.save(os.path.join(args.model_dir, 'seq2seq_model'))
    # save tokenizers and metadata
    with open(os.path.join(args.model_dir, 'eng_tokenizer.pkl'), 'wb') as f:
        pickle.dump(eng_tok, f)
    with open(os.path.join(args.model_dir, 'fra_tokenizer.pkl'), 'wb') as f:
        pickle.dump(fra_tok, f)
    meta = {'max_enc_len': args.max_enc_len, 'max_dec_len': args.max_dec_len}
    with open(os.path.join(args.model_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        import json
        json.dump(meta, f)
    print('Training artifacts saved to', args.model_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/sample_eng_fra.txt')
    parser.add_argument('--model_dir', default='models')
    parser.add_argument('--num_words', type=int, default=10000)
    parser.add_argument('--max_enc_len', type=int, default=20)
    parser.add_argument('--max_dec_len', type=int, default=20)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_examples', type=int, default=None)
    args = parser.parse_args()
    main(args)
