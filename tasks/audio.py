from __future__ import division
from percept.tasks.base import Task
from percept.tasks.train import Train
from percept.fields.base import Complex, List, Dict, Float
from inputs.inputs import SimpsonsFormats
from percept.utils.models import RegistryCategories, get_namespace
import logging
from percept.tests.framework import Tester
from percept.conf.base import settings
import re
import os
from matplotlib import pyplot
import numpy as np
from scikits.audiolab import Sndfile
from scikits.audiolab import oggread
import pandas as pd
from multiprocessing import Pool, TimeoutError
from sklearn.ensemble import RandomForestClassifier
import math
import random
from itertools import chain

log = logging.getLogger(__name__)

class LoadAudioFiles(Task):
    data = Complex()
    all_files = List()
    seq = Complex()
    res = Complex()
    label_codes = Dict()

    data_format = SimpsonsFormats.dataframe

    category = RegistryCategories.preprocessors
    namespace = get_namespace(__module__)

    help_text = "Cleanup simpsons scripts."

    args = {
        'audio_dir' : settings.AUDIO_DIR,
        'timeout' : 600,
        'only_labelled_lines' : settings.ONLY_LABELLED_LINES,
        'processed_files_limit' : settings.PROCESSED_FILES_LIMIT
    }

    def train(self, data, target, **kwargs):
        """
        Used in the training phase.  Override.
        """
        self.data = self.predict(data, **kwargs)

    def extract_season(self,name):
        match1 = re.search('\[(\d+)[x\.](\d+)\]',name)
        if match1 is not None:
            season = match1.group(1)
            episode = match1.group(2)
            return int(season),int(episode)

        match2 = re.search('S(\d+)E(\d+)',name)
        if match2 is not None:
            season = match2.group(1)
            episode = match2.group(2)
            return int(season),int(episode)

        return None, None

    def predict(self, data, **kwargs):
        """
        Used in the predict phase, after training.  Override
        """
        p = Pool(4, maxtasksperchild=50)
        audio_dir = kwargs['audio_dir']
        timeout = kwargs['timeout']
        oll = kwargs['only_labelled_lines']
        pff = kwargs['processed_files_limit']

        all_files = []
        for ad in os.listdir(audio_dir):
            ad_path = os.path.abspath(os.path.join(audio_dir,ad))
            if os.path.isdir(ad_path):
                files = os.listdir(ad_path)
                all_files += [os.path.abspath(os.path.join(ad_path,f)) for f in files]
            else:
                all_files += [ad_path]
        self.all_files = [f for f in all_files if f.endswith(".ogg")]
        frames = []
        counter = 0
        for f in self.all_files:
            season,episode = self.extract_season(f)
            if season is None or (season==11 and episode==6):
                continue
            subtitle_frame = data[((data['season']==season) & (data['episode']==episode))]
            if subtitle_frame.shape[0]==0:
                continue

            #To cause loop to end early, remove if needed
            if oll:
                label_frame = subtitle_frame[(subtitle_frame['label']!="")]
                if label_frame.shape[0]==0:
                    continue
            if pff is not None and isinstance(pff, int) and counter>=pff:
                break

            counter+=1
            log.info("On file {0} Season {1} Episode {2}".format(counter,season,episode))
            f_data, fs, enc  = oggread(f)
            subtitle_frame = subtitle_frame.sort('start')
            subtitle_frame.index = range(subtitle_frame.shape[0])
            samps = []
            good_rows = []
            for i in xrange(0,subtitle_frame.shape[0]):
                start = subtitle_frame['start'].iloc[i]
                end = subtitle_frame['end'].iloc[i]
                if end-start>6 or (subtitle_frame['label'][i]=='' and oll):
                    continue
                samp = f_data[(start*fs):(end*fs),:]
                samps.append({'samp' : samp, 'fs' : fs})
                good_rows.append(i)
            r = p.imap(process_subtitle, samps,chunksize=1)
            sf = subtitle_frame.iloc[good_rows]
            results = []
            for i in range(len(samps)):
                try:
                    results.append(r.next(timeout=timeout))
                except TimeoutError:
                    results.append(None)
            good_rows = [i for i in xrange(0,len(results)) if results[i]!=None]
            audio_features = [i for i in results if i!=None]
            good_sf = sf.iloc[good_rows]
            good_sf.index = range(good_sf.shape[0])
            audio_frame = pd.DataFrame(audio_features)
            audio_frame.index = range(audio_frame.shape[0])
            df = pd.concat([good_sf,audio_frame],axis=1)
            df = df.fillna(-1)
            df.index = range(df.shape[0])
            frames.append(df)
            lab_df_shape = df[df['label']!=''].shape[0]
            log.info("Processed {0} lines, {1} of which were labelled".format(df.shape[0],lab_df_shape))
        p.close()
        p.join()
        log.info("Done processing episodes.")
        data = pd.concat(frames,axis=0)
        data.index = range(data.shape[0])
        data.index = range(data.shape[0])

        for c in list(data.columns):
            data[c] = data[c].real
        for k in CHARACTERS:
            for i in CHARACTERS[k]:
                data['label'][data['label']==i] = k
        self.label_codes = {k:i for (i,k) in enumerate(set(data['label']))}
        reverse_label_codes = {self.label_codes[k]:k for k in self.label_codes}
        data['label_code'] = [self.label_codes[k] for k in data['label']]
        self.seq = SequentialValidate()

        #Do cv to get error estimates
        cv_frame = data[data['label']!=""]
        self.seq.train(cv_frame,**self.seq.args)
        self.res = self.seq.results
        self.res = self.res[['line', 'label','label_code','result_code','result_label']]

        exact_percent, adj_percent = compute_error(self.res)
        log.info("Exact match percent: {0}".format(exact_percent))
        log.info("Adjacent match percent: {0}".format(adj_percent))
        #Predict in the frame
        alg = RandomForestTrain()
        target = cv_frame['label_code']
        non_predictors = ["label","line","label_code"]
        train_names = [l for l in list(cv_frame.columns) if l not in non_predictors]
        train_data = cv_frame[train_names]
        predict_data = data[train_names]
        clf = alg.train(train_data,target,**alg.args)
        data['result_code'] = alg.predict(predict_data)
        data['result_label'] = [reverse_label_codes[k] for k in data['result_code']]
        return data

def compute_error(data):
    exact_match = data[data['result_code']==data['label_code']]
    exact_match_percent = exact_match.shape[0]/data.shape[0]

    adjacent_match = []
    for i in xrange(0,data.shape[0]):
        start = i-1
        if start<1:
            start = 1
        end = i+2
        if end>data.shape[0]:
            end = data.shape[0]
        sel_labs = list(data.iloc[start:end]['label_code'])
        adjacent_match.append(data['result_code'][i] in sel_labs)
    adj_percent = sum(adjacent_match)/data.shape[0]

    return exact_match_percent,adj_percent

def calc_slope(x,y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_dev = np.sum(np.abs(np.subtract(x,x_mean)))
    y_dev = np.sum(np.abs(np.subtract(y,y_mean)))

    slope = (x_dev*y_dev)/(x_dev*x_dev)
    return slope

def get_indicators(vec):
    mean = np.mean(vec)
    slope = calc_slope(np.arange(len(vec)),vec)
    std = np.std(vec)
    return mean, slope, std

def calc_u(vec):
    fft = np.fft.fft(vec)
    return np.sum(np.multiply(fft,vec))/np.sum(vec)

def calc_features(vec,freq):
    #bin count
    bc = 10
    bincount = list(range(bc))
    #framesize
    fsize = 512
    #mean
    m = np.mean(vec)
    #spectral flux
    sf = np.mean(vec-np.roll(vec,fsize))
    mx = np.max(vec)
    mi = np.min(vec)
    sdev = np.std(vec)
    binwidth = len(vec)/bc
    bins = []
    for i in xrange(0,bc):
        bins.append(vec[(i*binwidth):(binwidth*i + binwidth)])
    peaks = [np.max(i) for i in bins]
    mins = [np.min(i) for i in bins]
    amin,smin,stmin = get_indicators(mins)
    apeak, speak, stpeak = get_indicators(peaks)
    #fft = np.fft.fft(vec)
    bin_fft = []
    for i in xrange(0,bc):
        bin_fft.append(np.fft.fft(vec[(i*binwidth):(binwidth*i + binwidth)]))

    cepstrums = [np.fft.ifft(np.log(np.abs(i))) for i in bin_fft]
    inter = [get_indicators(i) for i in cepstrums]
    acep,scep, stcep = get_indicators([i[0] for i in inter])
    aacep,sscep, stsscep = get_indicators([i[1] for i in inter])

    zero_crossings = np.where(np.diff(np.sign(vec)))[0]
    zcc = len(zero_crossings)
    zccn = zcc/freq

    u = [calc_u(i) for i in bins]
    spread = np.sqrt(u[-1] - u[0]**2)
    skewness = (u[0]**3 - 3*u[0]*u[5] + u[-1])/spread**3

    #Spectral slope
    #ss = calc_slope(np.arange(len(fft)),fft)
    avss = [calc_slope(np.arange(len(i)),i) for i in bin_fft]
    savss = calc_slope(bincount,avss)
    mavss = np.mean(avss)

    return [m,sf,mx,mi,sdev,amin,smin,stmin,apeak,speak,stpeak,acep,scep,stcep,aacep,sscep,stsscep,zcc,zccn,spread,skewness,savss,mavss]

def extract_features(sample,freq):
    left = calc_features(sample[:,0],freq)
    right = calc_features(sample[:,1],freq)
    return left+right

def process_subtitle(d):
    samp = d['samp']
    fs = d['fs']
    if isinstance(samp,basestring):
        return None
    try:
        features = extract_features(samp,fs)
    except Exception:
        log.exception("Cannot generate features")
        return None

    return features

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

class CrossValidate(Task):
    data = Complex()
    results = Complex()
    error = Float()
    importances = Complex()
    importance = Complex()
    column_names = List()

    data_format = SimpsonsFormats.dataframe

    category = RegistryCategories.preprocessors
    namespace = get_namespace(__module__)
    args = {
        'nfolds' : 3,
        'algo' : RandomForestTrain,
        'target_name' : 'label_code',
        'non_predictors' : ["label","line","label_code"]
    }

    help_text = "Cross validate simpsons data."

    def cross_validate(self, data, **kwargs):
        nfolds = kwargs.get('nfolds', 3)
        algo = kwargs.get('algo')
        seed = kwargs.get('seed', 1)
        self.target_name = kwargs.get('target_name')
        non_predictors = kwargs.get('non_predictors')

        self.column_names = [l for l in list(data.columns) if l not in non_predictors]
        data_len = data.shape[0]
        counter = 0
        fold_length = int(math.floor(data_len/nfolds))
        folds = []
        data_seq = list(xrange(0,data_len))
        random.seed(seed)
        random.shuffle(data_seq)

        for fold in xrange(0, nfolds):
            start = counter

            end = counter + fold_length
            if fold == (nfolds-1):
                end = data_len
            folds.append(data_seq[start:end])
            counter += fold_length

        results = []
        data.index = range(data.shape[0])
        self.importances = []
        for (i,fold) in enumerate(folds):
            predict_data = data.iloc[fold,:]
            out_indices = list(chain.from_iterable(folds[:i] + folds[(i + 1):]))
            train_data = data.iloc[out_indices,:]
            alg = algo()
            target = train_data[self.target_name]
            train_data = train_data[[l for l in list(train_data.columns) if l not in non_predictors]]
            predict_data = predict_data[[l for l in list(predict_data.columns) if l not in non_predictors]]
            clf = alg.train(train_data,target,**algo.args)
            results.append(alg.predict(predict_data))
            self.importances.append(clf.feature_importances_)
        return results, folds

    def train(self, data, target, **kwargs):
        """
        Used in the training phase.  Override.
        """
        self.target_name = kwargs.get('target_name')
        results, folds = self.cross_validate(data, **kwargs)
        self.gather_results(results, folds, data)

    def gather_results(self, results, folds, data):
        full_results = list(chain.from_iterable(results))
        full_indices = list(chain.from_iterable(folds))
        partial_result_df = make_df([full_results, full_indices], ["result", "index"])
        partial_result_df = partial_result_df.sort(["index"])
        partial_result_df.index = range(partial_result_df.shape[0])
        result_df = pd.concat([partial_result_df, data], axis=1)
        self.results = result_df
        self.calc_importance(self.importances, self.column_names)

    def calc_error(self, result_df):
        self.error = np.mean(np.abs(result_df['result'] - result_df[self.target_name]))

    def calc_importance(self, importances, col_names):
        importance_frame = pd.DataFrame(importances)
        importance_frame.columns = col_names
        self.importance = importance_frame.mean(axis=0)
        self.importance.sort(0)

    def predict(self, data, **kwargs):
        """
        Used in the predict phase, after training.  Override
        """
        pass

def make_df(datalist, labels, name_prefix=""):
    df = pd.DataFrame(datalist).T
    if name_prefix!="":
        labels = [name_prefix + "_" + l for l in labels]
    labels = [l.replace(" ", "_").lower() for l in labels]
    df.columns = labels
    df.index = range(df.shape[0])
    return df


CHARACTERS = {
    'Tertiary': [
                 'Willy',
                 'Hibbert',
                 'Ralph',
                 'Barney',
                 'Carl',
                 'Otto',
                 'Dr.Nick',
                 'Ms.K',
                 'Teacher',
                 'Kids',
                 'Santa',
                 'Lenny',
                 'Comic Book Guy',
                 'Quimby',
                 'Ms.Hoover',
                 'Patty',
                 'Duffman',
                 'Troy',
                 'Kid'],
    }

"""
from tasks.train import Vectorizer
v = Vectorizer()


log.info(data['label'])
v.fit(list(data['line']),list(data['label_code']))
feats = v.batch_get_features(list(data['line']))
feats_frame = pd.DataFrame(feats)
feats_frame.columns = list(xrange(100,feats_frame.shape[1]+100))
feats_frame.index = range(feats_frame.range[0])
data = pd.concat([data,feats_frame],axis=1)
data = data.fillna(-1)
"""

class SequentialValidate(CrossValidate):
    args = {
        'min_years' : 10,
        'algo' : RandomForestTrain,
        'split_var' : 'season',
        'target_name' : 'label_code',
        'non_predictors' : ["label","line","label_code", 'result_label','result_code']
    }
    def sequential_validate(self, data, **kwargs):
        algo = kwargs.get('algo')
        seed = kwargs.get('seed', 1)
        split_var = kwargs.get('split_var')
        non_predictors = kwargs.get('non_predictors')
        self.target_name = kwargs.get('target_name')
        random.seed(seed)
        label_codes = {k:i for (i,k) in enumerate(set(data['label']))}
        results = []
        self.importances = []
        unique_seasons = list(set(data[split_var]))
        for s in unique_seasons:
            train_data = data[data[split_var] != s]
            predict_full = data[data[split_var] == s]

            alg = algo()

            target = train_data[self.target_name]
            train_names = [l for l in list(train_data.columns) if l not in non_predictors]
            train_data = train_data[train_names]
            predict_data = predict_full[train_names]

            clf = alg.train(train_data,target, **algo.args)
            predict_full['result_code'] = alg.predict(predict_data)
            predict_full['confidence'] = np.amax(clf.predict_proba(predict_data))
            self.importances.append(clf.feature_importances_)
            results.append(predict_full)

        reverse_label_codes = {label_codes[k]:k for k in label_codes}
        reverse_label_codes.update({-1 : ''})
        self.results = pd.concat(results,axis=0,ignore_index=True)
        self.results['result_label'] = [reverse_label_codes[k] for k in self.results['result_code']]

        self.calc_importance(self.importances, train_names)

    def train(self, data, **kwargs):
        """
        Used in the training phase.  Override.
        """
        self.sequential_validate(data, **kwargs)