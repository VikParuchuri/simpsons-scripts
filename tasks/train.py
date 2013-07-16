from __future__ import division
from itertools import chain
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from utils import get_message_replies, MessageReply, get_submission_reply_pairs, read_raw_data_from_cache, write_data_to_cache
from scipy.spatial.distance import euclidean
from fisher import pvalue
import random
import settings
import re
import collections
from nltk.stem.porter import PorterStemmer
import math

MAX_FEATURES = 100

class SpellCorrector(object):
    """
    Taken and slightly adapted from peter norvig's post at http://norvig.com/spell-correct.html
    """

    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    punctuation = [".", "!", "?", ","]

    def __init__(self):
        self.NWORDS = self.train(self.words(file('big.txt').read()))

    def words(self, text):
        return re.findall('[a-z]+', text.lower())

    def train(self, features):
        model = collections.defaultdict(lambda: 1)
        for f in features:
            model[f] += 1
        return model

    def edits1(self, word):
        splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes    = [a + b[1:] for a, b in splits if b]
        transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
        replaces   = [a + c + b[1:] for a, b in splits for c in self.alphabet if b]
        inserts    = [a + c + b     for a, b in splits for c in self.alphabet]
        return set(deletes + transposes + replaces + inserts)

    def known_edits2(self, word):
        return set(e2 for e1 in self.edits1(word) for e2 in self.edits1(e1) if e2 in self.NWORDS)

    def known(self, words): return set(w for w in words if w in self.NWORDS)

    def correct(self, word):
        suffix = ""
        for p in self.punctuation:
            if word.endswith(p):
                suffix = p
                word = word[:-1]
        candidates = self.known([word]) or self.known(self.edits1(word)) or self.known_edits2(word) or [word]
        word = max(candidates, key=self.NWORDS.get)
        return word + suffix

class Vectorizer(object):
    def __init__(self):
        self.fit_done = False

    def fit(self, input_text, input_scores, max_features=100):
        self.spell_corrector = SpellCorrector()
        self.stemmer = PorterStemmer()
        new_text = self.batch_generate_new_text(input_text)
        input_text = [input_text[i] + new_text[i] for i in xrange(0,len(input_text))]
        self.vectorizer1 = CountVectorizer(ngram_range=(1,2), min_df = 3/len(input_text), max_df=.4)
        self.vectorizer1.fit(input_text)
        self.vocab = self.get_vocab(input_text, input_scores, max_features)
        self.vectorizer = CountVectorizer(ngram_range=(1,2), vocabulary=self.vocab)
        self.fit_done = True
        self.input_text = input_text

    def spell_correct_text(self, text):
        text = text.lower()
        split = text.split(" ")
        corrected = [self.spell_corrector.correct(w) for w in split]
        return corrected

    def batch_apply(self, all_tokens, applied_func):
        for key in all_tokens:
            cor = applied_func(all_tokens[key])
            all_tokens[key] = cor
        return all_tokens

    def batch_generate_new_text(self, text):
        text = [re.sub("[^A-Za-z0-9]", " ", t.lower()) for t in text]
        text = [re.sub("\s+", " ", t) for t in text]
        t_tokens = [t.split(" ") for t in text]
        all_token_list = list(set(chain.from_iterable(t_tokens)))
        all_token_dict = {}
        for t in all_token_list:
            all_token_dict.update({t : t})
        all_token_dict = self.batch_apply(all_token_dict, self.stemmer.stem)
        all_token_dict = self.batch_apply(all_token_dict, self.stemmer.stem)
        for i in xrange(0,len(t_tokens)):
            for j in xrange(0,len(t_tokens[i])):
                t_tokens[i][j] = all_token_dict.get(t_tokens[i][j], t_tokens[i][j])
        new_text = [" ".join(t) for t in t_tokens]
        return new_text

    def generate_new_text(self, text):
        no_punctuation = re.sub("[^A-Za-z0-9]", " ", text.lower())
        no_punctuation = re.sub("\s+", " ", no_punctuation)
        corrected = self.spell_correct_text(no_punctuation)
        corrected = [self.stemmer.stem(w) for w in corrected]
        new = " ".join(corrected)
        return new

    def get_vocab(self, input_text, input_scores, max_features):
        train_mat = self.vectorizer1.transform(input_text)
        input_score_med = np.median(input_scores)
        new_scores = [0 if i<=input_score_med else 1 for i in input_scores]
        pvalues = []
        for i in xrange(0,train_mat.shape[1]):
            lcol = np.asarray(train_mat.getcol(i).todense().transpose())[0]
            good_lcol = lcol[[n for n in xrange(0,len(new_scores)) if new_scores[n]==1]]
            bad_lcol = lcol[[n for n in xrange(0,len(new_scores)) if new_scores[n]==0]]
            good_lcol_present = len(good_lcol[good_lcol > 0])
            good_lcol_missing = len(good_lcol[good_lcol == 0])
            bad_lcol_present = len(bad_lcol[bad_lcol > 0])
            bad_lcol_missing = len(bad_lcol[bad_lcol == 0])
            pval = pvalue(good_lcol_present, bad_lcol_present, good_lcol_missing, bad_lcol_missing)
            pvalues.append(pval.two_tail)
        col_inds = list(xrange(0,train_mat.shape[1]))
        p_frame = pd.DataFrame(np.array([col_inds, pvalues]).transpose(), columns=["inds", "pvalues"])
        p_frame.sort(['pvalues'], ascending=True)
        getVar = lambda searchList, ind: [searchList[int(i)] for i in ind]
        vocab = getVar(self.vectorizer1.get_feature_names(), p_frame['inds'][:max_features])
        return vocab

    def batch_get_features(self, text):
        if not self.fit_done:
            raise Exception("Vectorizer has not been created.")
        new_text = self.batch_generate_new_text(text)
        text = [text[i] + new_text[i] for i in xrange(0,len(text))]
        return (self.vectorizer.transform(text).todense())

    def get_features(self, text):
        if not self.fit_done:
            raise Exception("Vectorizer has not been created.")
        itext=text
        if isinstance(text, list):
            itext = text[0]
        new_text = self.generate_new_text(itext)
        if isinstance(text, list):
            text = [text[0] + new_text]
        else:
            text = text + new_text
        return (self.vectorizer.transform(text).todense())

class KNNCommentMatcher(object):
    def __init__(self, train_data):
        self.train = train_data
        self.max_features = math.floor(MAX_FEATURES)/3
        self.vectorizer = Vectorizer()
        self.lines = [t['line'] for t in self.train]
        self.speaker_codes = [t['speaker_code'] for t in self.train]

    def fit(self):
        self.vectorizer.fit(self.lines, self.speaker_codes, self.max_features)
        self.train_mat = self.vectorizer.batch_get_features(self.lines)

    def find_nearest_match(self, text):
        test_vec = np.asarray(self.vectorizer.get_features(text))
        distances = [euclidean(u, test_vec) for u in self.train_mat]
        nearest_match = distances.index(min(distances))
        return nearest_match, min(distances)

    def find_knn_speaker(self, text):
        nearest_match, distance = self.find_nearest_match(text)
        raw_data = self.train[nearest_match]
        return raw_data, nearest_match, distance
