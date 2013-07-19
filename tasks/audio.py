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
from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier
import math
import random
from itertools import chain

log = logging.getLogger(__name__)

class LoadAudioFiles(Task):
    data = Complex()
    all_files = List()
    cv = Complex()
    res = Complex()

    data_format = SimpsonsFormats.dataframe

    category = RegistryCategories.preprocessors
    namespace = get_namespace(__module__)

    help_text = "Cleanup simpsons scripts."

    args = {
        'audio_dir' : os.path.abspath(os.path.join(settings.AUDIO_BASE_PATH, "audio")),
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
        audio_dir = kwargs['audio_dir']
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
            if season is None:
                continue
            subtitle_frame = data[((data['season']==season) & (data['episode']==episode))]
            if subtitle_frame.shape[0]==0:
                continue

            #To cause loop to end early, remove if needed
            label_frame = subtitle_frame[(subtitle_frame['label']!="")]
            if label_frame.shape[0]==0:
                continue
            if counter>5:
                break

            counter+=1
            print "On file {0}".format(counter)
            f_data, fs, enc  = oggread(f)
            subtitle_frame = subtitle_frame.sort('start')
            subtitle_frame.index = range(subtitle_frame.shape[0])
            samps = []
            for i in xrange(0,subtitle_frame.shape[0]):
                start = subtitle_frame['start'].iloc[i]
                end = subtitle_frame['end'].iloc[i]
                samp = f_data[(start*fs):(end*fs),:]
                samps.append({'samp' : samp, 'fs' : fs})
            p = Pool(4)
            results = p.map(process_subtitle, samps)
            good_rows = [i for i in xrange(0,len(results)) if results[i]!=None]
            audio_features = [i for i in results if i!=None]
            df = pd.concat([subtitle_frame.iloc[good_rows],pd.DataFrame(audio_features)],axis=1)
            df = df.fillna(-1)
            df.index = range(df.shape[0])
            frames.append(df)
        data = pd.concat(frames,axis=0)
        data.index = range(data.shape[0])
        label_codes = {k:i for (i,k) in enumerate(set(data['label']))}
        reverse_label_codes = {label_codes[k]:k for k in label_codes}
        data['label_code'] = [label_codes[k] for k in data['label']]
        for c in list(data.columns):
            data[c] = data[c].real
        self.cv = CrossValidate()

        cv_frame = data[data['label']!=""]
        self.cv.train(cv_frame,"",**self.cv.args)
        self.res = self.cv.results
        self.res = self.res[['line', 'label','label_code','result']]
        self.res['actual_result'] = [reverse_label_codes[i] for i in self.res['result']]
        return data

def calc_slope(x,y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_dev = np.sum(np.abs(np.subtract(x,x_mean)))
    y_dev = np.sum(np.abs(np.subtract(y,y_mean)))

    slope = (x_dev*y_dev)/(x_dev*x_dev)
    return slope

def get_indicators(vec):
    mean = np.sum(vec)
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
    fft = np.fft.fft(vec)
    bin_fft = []
    for i in xrange(0,bc):
        bin_fft.append(fft[(i*binwidth):(binwidth*i + binwidth)])

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
    ss = calc_slope(np.arange(len(fft)),fft)
    avss = calc_slope(bincount,[calc_slope(np.arange(len(i)),i) for i in bin_fft])

    return [m,sf,mx,mi,sdev,amin,smin,stmin,apeak,speak,stpeak,acep,scep,stcep,aacep,sscep,stsscep,zcc,zccn,spread,skewness,ss,avss]

def extract_features(sample,freq):
    left = calc_features(sample[:,0],freq)
    right = calc_features(sample[:,1],freq)
    return left+right

def process_subtitle(d):
    samp = d['samp']
    fs = d['fs']
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
        target_name = kwargs.get('target_name')
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
            target = train_data[target_name]
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
        self.calc_error(result_df)
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