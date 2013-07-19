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
            return season,episode

        match2 = re.search('S(\d+)E(\d+)',name)
        if match2 is not None:
            season = match2.group(1)
            episode = match2.group(2)
            return season,episode

        return None, None

    def calc_slope(self,x,y):
        x_mean = sum(x)/len(x)
        y_mean = sum(y)/len(y)
        x_dev = sum([abs(i-x_mean) for i in x])
        y_dev = sum([abs(i-y_mean) for i in y])

        slope = (x_dev*y_dev)/(x_dev*x_dev)
        return slope

    def get_indicators(self,vec):
        mean = sum(vec)
        slope = self.calc_slope(list(range(len(vec))),vec)
        std = np.std(vec)
        return mean, slope, std

    def calc_features(self, vec,freq):
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
        amin,smin,stmin = self.get_indicators(mins)
        apeak, speak, stpeak = self.get_indicators(peaks)
        cepstrums = [np.fft.ifft(np.log(np.abs(np.fft.fft(i)))) for i in bins]
        inter = [self.get_indicators(i) for i in cepstrums]
        acep,scep, stcep = self.get_indicators([i[0] for i in inter])
        aacep,sscep, stsscep = self.get_indicators([i[1] for i in inter])

        zero_crossings = np.where(np.diff(np.sign(vec)))[0]
        zcc = len(zero_crossings)
        zccn = zcc/freq

        return [m,sf,mx,mi,sdev,amin,smin,stmin,apeak,speak,stpeak,acep,scep,stcep,aacep,sscep,stsscep,zcc,zccn]


    def extract_features(self,sample,freq):
        left = self.calc_features(sample[:,0],freq)
        right = self.calc_features(sample[:,1],freq)
        return left+right

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
        for f in self.all_files:
            season,episode = self.extract_season(f)
            if season is None:
                continue
            f_data, fs, enc  = oggread(f)
            subtitle_frame = data[((data['season']==season) & (data['episode']==episode))]
            if subtitle_frame.shape[0]==0:
                continue
            current_frame = 0
            subtitle_frame = subtitle_frame.sort('start')
            audio_samples = []
            audio_features = []
            for i in xrange(0,subtitle_frame.shape[0]):
                start = subtitle_frame['start'][i]
                end = subtitle_frame['end'][i]
                samp = f_data[(start*fs):(end*fs),:]
                features = self.extract_features(samp,fs)
                audio_features.append(features)
            df = pd.concat([subtitle_frame,pd.DataFrame(audio_features)],axis=1)
            df = df.fillna(-1)
            df.columns = range(df.shape[1])
            df.index = range(df.shape[0])
            frames.append(df)
        data = pd.concat(frames,axis=0)

        return data