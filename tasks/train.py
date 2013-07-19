from __future__ import division
from itertools import chain
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from fisher import pvalue
import re
import collections
from nltk.stem.porter import PorterStemmer
import math
from percept.tasks.base import Task
from percept.fields.base import Complex, List, Dict, Float
from inputs.inputs import SimpsonsFormats
from percept.utils.models import RegistryCategories, get_namespace
from percept.conf.base import settings
import os
from percept.tasks.train import Train
from sklearn.ensemble import RandomForestClassifier
import pickle
import random

import logging
log = logging.getLogger(__name__)

MAX_FEATURES = 500
DISTANCE_MIN=1
CHARACTER_DISTANCE_MIN = .2
RESET_SCENE_EVERY = 5

def make_df(datalist, labels, name_prefix=""):
    df = pd.DataFrame(datalist).T
    if name_prefix!="":
        labels = [name_prefix + "_" + l for l in labels]
    labels = [l.replace(" ", "_").lower() for l in labels]
    df.columns = labels
    df.index = range(df.shape[0])
    return df

def return_one():
    return 1

class SpellCorrector(object):
    """
    Taken and slightly adapted from peter norvig's post at http://norvig.com/spell-correct.html
    """

    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    punctuation = [".", "!", "?", ","]

    def __init__(self):
        self.NWORDS = self.train(self.words(file(os.path.join(settings.PROJECT_PATH,'data/big.txt')).read()))
        self.cache = {}

    def words(self, text):
        return re.findall('[a-z]+', text.lower())

    def train(self, features):
        model = collections.defaultdict(return_one)
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
        if word in self.cache:
            return self.cache[word]
        suffix = ""
        for p in self.punctuation:
            if word.endswith(p):
                suffix = p
                word = word[:-1]
        candidates = self.known([word]) or self.known(self.edits1(word)) or self.known_edits2(word) or [word]
        newword = max(candidates, key=self.NWORDS.get) + suffix
        self.cache.update({word : newword})
        return newword

class Vectorizer(object):
    def __init__(self):
        self.fit_done = False

    def fit(self, input_text, input_scores, max_features=100, min_features=3):
        self.spell_corrector = SpellCorrector()
        self.stemmer = PorterStemmer()
        new_text = self.batch_generate_new_text(input_text)
        input_text = [input_text[i] + new_text[i] for i in xrange(0,len(input_text))]
        self.vectorizer1 = CountVectorizer(ngram_range=(1,2), min_df = min_features/len(input_text), max_df=.4, stop_words="english")
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
        ind_max_features = math.floor(max_features/max(input_scores))
        all_vocab = []
        all_cols = [np.asarray(train_mat.getcol(i).todense().transpose())[0] for i in xrange(0,train_mat.shape[1])]
        for s in xrange(0,max(input_scores)):
            sel_inds = [i for i in xrange(0,len(input_scores)) if input_scores[i]==s]
            out_inds = [i for i in xrange(0,len(input_scores)) if input_scores[i]!=s]
            pvalues = []
            for i in xrange(0,len(all_cols)):
                lcol = all_cols[i]
                good_lcol = lcol[sel_inds]
                bad_lcol = lcol[out_inds]
                good_lcol_present = len(good_lcol[good_lcol > 0])
                good_lcol_missing = len(good_lcol[good_lcol == 0])
                bad_lcol_present = len(bad_lcol[bad_lcol > 0])
                bad_lcol_missing = len(bad_lcol[bad_lcol == 0])
                pval = pvalue(good_lcol_present, bad_lcol_present, good_lcol_missing, bad_lcol_missing)
                pvalues.append(pval.two_tail)
            col_inds = list(xrange(0,train_mat.shape[1]))
            p_frame = pd.DataFrame(np.array([col_inds, pvalues]).transpose(), columns=["inds", "pvalues"])
            p_frame = p_frame.sort(['pvalues'], ascending=True)
            getVar = lambda searchList, ind: [searchList[int(i)] for i in ind]
            vocab = getVar(self.vectorizer1.get_feature_names(), p_frame['inds'][:ind_max_features+2])
            all_vocab.append(vocab)
        return list(set(list(chain.from_iterable(all_vocab))))

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
            text = [text + new_text]
        return (self.vectorizer.transform(text).todense())

class FeatureExtractor(Task):
    data = Complex()
    row_data = List()
    speaker_code_dict = Dict()
    speaker_codes = List()
    vectorizer = Complex()

    data_format = SimpsonsFormats.dataframe

    category = RegistryCategories.preprocessors
    namespace = get_namespace(__module__)

    help_text = "Cleanup simpsons scripts."

    args = {'scriptfile' : os.path.abspath(os.path.join(settings.DATA_PATH, "script_tasks"))}

    def train(self, data, target, **kwargs):
        """
        Used in the training phase.  Override.
        """
        self.data = self.predict(data, **kwargs)

    def predict(self, data, **kwargs):
        """
        Used in the predict phase, after training.  Override
        """
        scriptfile = kwargs.get('scriptfile')
        script_data = pickle.load(open(scriptfile))
        script = script_data.tasks[2].voice_lines.value
        speakers = []
        lines = []
        for s in script:
            for (i,l) in enumerate(s):
                if i>0:
                    previous_line = s[i-1]['line']
                    previous_speaker = s[i-1]['speaker']
                else:
                    previous_line = ""
                    previous_speaker = ""

                if i>1:
                    two_back_speaker = s[i-2]['speaker']
                else:
                    two_back_speaker = ""

                if len(s)>i+1:
                    next_line = s[i+1]['line']
                else:
                    next_line = ""
                current_line = s[i]['line']
                current_speaker = s[i]['speaker']
                lines.append(current_line)
                speakers.append(current_speaker)
                row_data = {
                    'previous_line' : previous_line,
                    'previous_speaker' : previous_speaker,
                    'next_line' : next_line,
                    'current_line' : current_line,
                    'current_speaker' : current_speaker,
                    'two_back_speaker' : two_back_speaker
                }
                self.row_data.append(row_data)
        self.speaker_code_dict = {k:i for (i,k) in enumerate(list(set(speakers)))}
        self.speaker_codes = [self.speaker_code_dict[s] for s in speakers]
        self.max_features = math.floor(MAX_FEATURES)/3
        self.vectorizer = Vectorizer()
        self.vectorizer.fit(lines, self.speaker_codes, self.max_features)
        prev_features = self.vectorizer.batch_get_features([rd['previous_line'] for rd in self.row_data])
        cur_features = self.vectorizer.batch_get_features([rd['current_line'] for rd in self.row_data])
        next_features = self.vectorizer.batch_get_features([rd['next_line'] for rd in self.row_data])

        self.speaker_code_dict.update({'' : -1})
        meta_features = make_df([[self.speaker_code_dict[s['two_back_speaker']] for s in self.row_data], [self.speaker_code_dict[s['previous_speaker']] for s in self.row_data], self.speaker_codes],["two_back_speaker", "previous_speaker", "current_speaker"])
        #meta_features = make_df([[self.speaker_code_dict[s['two_back_speaker']] for s in self.row_data], self.speaker_codes],["two_back_speaker", "current_speaker"])
        train_frame = pd.concat([pd.DataFrame(prev_features),pd.DataFrame(cur_features),pd.DataFrame(next_features),meta_features],axis=1)
        train_frame.index = range(train_frame.shape[0])
        data = {
            'vectorizer' : self.vectorizer,
            'speaker_code_dict' : self.speaker_code_dict,
            'train_frame' : train_frame,
            'speakers' : make_df([speakers,self.speaker_codes, lines], ["speaker", "speaker_code", "line"]),
            'data' : data,
            'current_features' : cur_features,
        }
        return data

class RandomForestTrain(Train):
    """
    A class to train a random forest
    """
    colnames = List()
    clf = Complex()
    category = RegistryCategories.algorithms
    namespace = get_namespace(__module__)
    algorithm = RandomForestClassifier
    args = {'n_estimators' : 300, 'min_samples_leaf' : 4, 'compute_importances' : True}

    help_text = "Train and predict with Random Forest."

class KNNRF(Task):
    data = Complex()
    predictions = Complex()
    importances = Complex()

    data_format = SimpsonsFormats.dataframe

    category = RegistryCategories.preprocessors
    namespace = get_namespace(__module__)

    args = {'algo' : RandomForestTrain}

    help_text = "Cleanup simpsons scripts."

    def train(self, data, target, **kwargs):
        """
        Used in the training phase.  Override.
        """
        self.data = self.predict(data, **kwargs)

    def predict(self, data, **kwargs):
        """
        Used in the predict phase, after training.  Override
        """
        from preprocess import CHARACTERS

        vec_length = math.floor(MAX_FEATURES/3)

        algo = kwargs.get('algo')
        alg = algo()
        train_data = data['train_frame'].iloc[:,:-1]
        target = data['train_frame']['current_speaker']
        clf = alg.train(train_data,target, **algo.args)
        self.importances=clf.feature_importances_

        test_data = data['data']
        match_data = data['current_features']
        reverse_speaker_code_dict = {data['speaker_code_dict'][k] : k for k in data['speaker_code_dict']}

        speaker_list = []
        speaker_codes = reverse_speaker_code_dict.keys()
        for i in xrange(0,len(speaker_codes)):
            s_text = "\n".join(list(data['speakers'][data['speakers']['speaker']==reverse_speaker_code_dict[speaker_codes[i]]]['line']))
            speaker_list.append(s_text)
        speaker_features = data['vectorizer'].batch_get_features(speaker_list)

        self.predictions = []
        counter = 0
        for script in test_data['voice_script']:
            counter+=1
            log.info("On script {0} out of {1}".format(counter,len(test_data['voice_script'])))
            lines = script.split("\n")
            speaker_code = [-1 for i in xrange(0,len(lines))]
            for (i,line) in enumerate(lines):
                if i>0 and i%RESET_SCENE_EVERY!=0:
                    previous_line = lines[i-1]
                    previous_speaker = speaker_code[i-1]
                else:
                    previous_line = ""
                    previous_speaker=  -1

                if i>1 and i%RESET_SCENE_EVERY!=0:
                    two_back_speaker = speaker_code[i-2]
                else:
                    two_back_speaker = -1

                if i<(len(lines)-1):
                    next_line = lines[i+1]
                else:
                    next_line = ""

                prev_features = data['vectorizer'].get_features(previous_line)
                cur_features = data['vectorizer'].get_features(line)
                next_features = data['vectorizer'].get_features(next_line)

                meta_features = make_df([[two_back_speaker], [previous_speaker]],["two_back_speaker", "previous_speaker"])
                #meta_features = make_df([[two_back_speaker]],["two_back_speaker"])
                train_frame = pd.concat([pd.DataFrame(prev_features),pd.DataFrame(cur_features),pd.DataFrame(next_features), meta_features],axis=1)

                speaker_code[i] = alg.predict(train_frame)[0]

                nearest_match, distance = self.find_nearest_match(cur_features, speaker_features)
                if distance<CHARACTER_DISTANCE_MIN:
                    sc = speaker_codes[nearest_match]
                    speaker_code[i] = sc
                    continue

                for k in CHARACTERS:
                    for c in CHARACTERS[k]:
                        if c in previous_line:
                            speaker_code[i] = data['speaker_code_dict'][k]

                nearest_match, distance = self.find_nearest_match(cur_features,match_data)
                if distance<DISTANCE_MIN:
                    sc = data['speakers']['speaker_code'][nearest_match]
                    speaker_code[i] = sc
                    continue

            df = make_df([lines,speaker_code,[reverse_speaker_code_dict[round(s)] for s in speaker_code]],["line","speaker_code","speaker"])
            self.predictions.append(df)
        return data

    def find_nearest_match(self, features, matrix):
        features = np.asarray(features)
        distances = [self.euclidean(u, features) for u in matrix]
        nearest_match = distances.index(min(distances))
        return nearest_match, min(distances)

    def euclidean(self, v1, v2):
        return np.sqrt(np.sum(np.square(np.subtract(v1,v2))))

"""
p = tasks[3].predictions.value
speakers = []
lines = []
for pr in p:
    speakers.append(list(pr['speaker']))
    lines.append(list(pr['line']))
from itertools import chain
speakers = list(chain.from_iterable(speakers))
lines = list(chain.from_iterable(lines))
rows = []
for (s,l) in zip(speakers, lines):
    rows.append({
    'speaker' : s,
    'line': l,
    })
import json
json.dump(rows,open("/home/vik/vikparuchuri/simpsons-scripts/data/final_voice.json","w+"))
"""