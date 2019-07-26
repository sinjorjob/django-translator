
import pandas as pd
import numpy as np
import tensorflow as tf
from dq_bot.function import to_sep_space, get_sim_index, show_sim_faq
from dq_bot.config import DQ_QA, DQ_CORPUS, DQ_MODEL
from dq_bot.net import AutoEncoder
from dq_bot.return_corpus import ReutersMuscleCorpus


class Infer():

    def __init__(self, encoder, corpus):
        self.encoder = encoder
        self.corpus = corpus

    def __call__(self, text, graph):
        text_sp = to_sep_space(text)
        ids = self.corpus.doc2ids(text_sp)
        vec = self.corpus.embed_matrix[ids]
        vec = np.reshape(vec, (1, vec.shape[0], vec.shape[1]))
        with graph.as_default():
            feat = self.encoder.predict(vec)
        return feat[0]