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

log = logging.getLogger(__name__)

class LoadAudioFiles(Task):
    data = Complex()
    all_files = List()

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

        return data

def calc_slope(x,y):
    x_mean = sum(x)/len(x)
    y_mean = sum(y)/len(y)
    x_dev = sum([abs(i-x_mean) for i in x])
    y_dev = sum([abs(i-y_mean) for i in y])

    slope = (x_dev*y_dev)/(x_dev*x_dev)
    return slope

def get_indicators(vec):
    mean = sum(vec)
    slope = calc_slope(list(range(len(vec))),vec)
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
    cepstrums = [np.fft.ifft(np.log(np.abs(np.fft.fft(i)))) for i in bins]
    inter = [get_indicators(i) for i in cepstrums]
    acep,scep, stcep = get_indicators([i[0] for i in inter])
    aacep,sscep, stsscep = get_indicators([i[1] for i in inter])

    zero_crossings = np.where(np.diff(np.sign(vec)))[0]
    zcc = len(zero_crossings)
    zccn = zcc/freq

    u = [calc_u(i) for i in bins]
    spread = np.sqrt(u[-1] - u[0]**2)
    skewness = (u[0]**3 - 3*u[0]*u[5] + u[-1])/spread**3

    return [m,sf,mx,mi,sdev,amin,smin,stmin,apeak,speak,stpeak,acep,scep,stcep,aacep,sscep,stsscep,zcc,zccn,spread,skewness]

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