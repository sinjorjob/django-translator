import pickle
import numpy as np
import pickle
from django.http.response import HttpResponse
from django.shortcuts import render, render_to_response
from django.contrib.staticfiles.templatetags.staticfiles import static
from . import forms
from django.template.context_processors import csrf
from dq_bot.function import load_models
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
seq_size = 15
batch_size = 4
n_epoch = 50
latent_size = 512


#from functions import get_logger, to_sep_space, get_sim_index, show_sim_faq
from dq_bot.config import DQ_QA, DQ_CORPUS, DQ_MODEL, JAWIKI_MODEL

from dq_bot.function import load_vectors, load_models
from dq_bot.return_corpus import ReutersMuscleCorpus
from dq_bot.dq_qa import Infer
import pickle
import numpy as np
import pandas as pd
import pickle
from dq_bot.return_corpus import ReutersMuscleCorpus


def make_pkl():
    WAKATI = MeCab.Tagger('-Ochasen')    #分かち書きの準備
    STOPWORDS = stopwords.words("english")  #stopワードの取得
    embed_matrix, vocab = load_vectors(JAWIKI_MODEL)
    corpus = ReutersMuscleCorpus()
    corpus.build(embed_matrix, vocab, seq_size)
    with open(DQ_CORPUS, 'wb') as f:
        pickle.dump(corpus, f)