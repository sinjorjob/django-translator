
import numpy as np
import tensorflow as tf
from keras.models import load_model
from dq_bot.net import AutoEncoder
from dq_bot.config import DQ_MODEL, PAD, UNK
from keras.backend import tensorflow_backend as backend
from sklearn.metrics.pairwise import cosine_similarity
import mojimoji
import io
import re
import MeCab


def load_models():
    backend.clear_session()
    model = AutoEncoder.load(DQ_MODEL)
    encoder, graph = model.get_encoder()
    #graph = tf.get_default_graph()
    return encoder, graph



def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    #データは「文字,分散表現（200）」の形式のデータになっている。
    #行数と分散ベクトルの次元数を取得
    a = list(map(int, fin.readline().split()))
    n = a[0]
    d = a[1]
    embed_matrix = np.zeros((n + 2, d))  # 語彙数はPAD, UNKを足す
    vocab = [PAD, UNK]
    i = 2
    for line in fin:
        #1行ごと読み込みスペースで分割して単語と分散ベクトルを取得
        tokens = line.rstrip().split(' ')
        # 語彙リストに単語を追加(0番目を削除して値を取り出す)
        word = tokens.pop(0)
        vocab.append(word)
        # tokensに残っている分散ベクトルをembed_matrixに追加
        v = np.asarray(tokens, dtype='float32')
        #0,1番目の要素は0（PAD,UNK)
        embed_matrix[i] = v
        i += 1
    return embed_matrix, vocab


def to_sep_space(txt):
    txt = mojimoji.zen_to_han(txt)  # 英数字は全て全角→半角に変換
    txt = re.sub(r'\d+', '0', txt)  # 連続した数字を0で置換
    parsed = WAKATI.parse(txt)
    #分かち書きした単語を連結して返す。
    sep_txt = [i.split('\t')[0] for i in parsed.split('\n') if i not in ['', 'EOS']]
    return ' '.join(sep_txt)


def to_sep_space(txt):
    txt = mojimoji.zen_to_han(txt)  # 英数字は全て全角→半角に変換
    txt = re.sub(r'\d+', '0', txt)  # 連続した数字を0で置換
    WAKATI = MeCab.Tagger('-Ochasen')    #分かち書きの準備
    parsed = WAKATI.parse(txt)
    
    #分かち書きした単語を連結して返す。
    sep_txt = [i.split('\t')[0] for i in parsed.split('\n') if i not in ['', 'EOS']]
    return ' '.join(sep_txt)


def get_sim_index(vec, vecs):
    sim = cosine_similarity(vec, vecs)
    sort_i = np.argsort(sim)[0][::-1]
    sim = np.sort(sim)[0][::-1]
    return sort_i, sim


def show_sim_faq(df, sim, n_top=3):
    index = 0
    for _, row in df.iterrows():
        print('-----------------------------')
        print('[{0}位] {1}: {2} ({3})'.format(index + 1, row['q_id'], row['q_txt'], sim[index]))
        print()
        print('{0}'.format(row['a_txt']))
        print('-----------------------------')
        index += 1
        if index >= n_top:
            break


def create_dict(i_chars, t_chars):
    #質問単語データの生成
    input_token_index = dict([
        (char, i) for i, char in enumerate(i_chars)
        ])
    #回答単語データの生成
    target_token_index = dict([
        (char, i) for i, char in enumerate(t_chars)
        ])
    return input_token_index, target_token_index

