import numpy as np
import pickle
from django.http.response import HttpResponse
from django.shortcuts import render, render_to_response
from django.contrib.staticfiles.templatetags.staticfiles import static
from . import forms
from django.template.context_processors import csrf
from dq_bot.function import load_models, get_sim_index
from keras.models import load_model
import six
import pandas as pd
import numpy as np
import mojimoji
import re
import os
import sys
import io
import re
import MeCab
import mojimoji
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
#from config import PAD, UNK, WIKIQA_DIR



#from functions import get_logger, to_sep_space, get_sim_index, show_sim_faq
from dq_bot.config import DQ_QA, DQ_CORPUS, DQ_MODEL, JAWIKI_MODEL

from dq_bot.function import load_vectors, load_models
from dq_bot.return_corpus import ReutersMuscleCorpus
from dq_bot.dq_qa import Infer
import pickle
import numpy as np
import pandas as pd
import pickle
#from functions import to_sep_space

from dq_bot.make_pkl import make_pkl

#モデル、GRAPHのロード
encoder, graph = load_models()
#pkl生成
#_ = make_pkl()


#コーパスファイルのロード
with open(DQ_CORPUS, 'rb') as f:
    corpus = pickle.load(f)

"""
各質問文に対して以下の処理を実行し、全質問文に対する予測結果（ベクトル表現）が得られる
INDEX化→ベクトル表現→モデルに突っ込んで予測→予測結果を得る。
"""
infer = Infer(encoder, corpus)

#QAデータのロード
qa_df = pd.read_csv(DQ_QA)

# モデルの予測結果をvecsに格納する。
q_txts = qa_df['q_txt'].tolist()



# 応答用の辞書を組み立てて返す
def __makedic(k,txt):
    return { 'k':k,'txt':txt}

def dq_bot(request):
    if request.method == 'POST':
        # テキストボックスに入力されたメッセージ
        input_seq = request.POST["messages"]
        form = forms.InputForm()
        # モデルの予測結果をvecsに格納する。
        vecs = np.array([infer(d, graph) for d in q_txts])
        #質問文をインプットして予測結果を取得
        vec = infer(input_seq, graph)
        
        #index(sort_i)と類似度(sim)が同じ順番で帰ってくる。
        sort_i, sim = get_sim_index([vec], vecs)
        df = qa_df.loc[sort_i]
        answers = []
        index = 0
        n_top=3
        for _, row in df.iterrows():
            answers.append([index + 1, row['q_txt'], row['a_txt'], sim[index]])
            index += 1
            if index >= n_top:
                break

        c = {
                 'form': form,
                 'answers':answers,
                }


        c.update(csrf(request))
        return render(request,'dq_bot/demo.html',c)


    else:
        # 初期表示の時にセッションもクリアする
        request.session.clear()
        # フォームの初期化
        form = forms.InputForm(label_suffix='：')

        c = {'form': form,
        }
    c.update(csrf(request))
    return render(request,'dq_bot/demo.html',c)


   