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
from sklearn.cluster import KMeans
import os
import json
from itertools import chain
from matplotlib import pyplot
import numpy as np
from sklearn.decomposition import PCA
from scikits.audiolab import Sndfile
from scikits.audiolab import oggread

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
        for f in self.all_files:
            season,episode = self.extract_season(f)
            if season is None:
                continue
            f_data, fs, enc  = oggread(f)
            subtitle_frame = data[((data['season']==season) & (data['episode']==episode))]
            if subtitle_frame.shape[0]==0:
                continue
            subtitle_frame = subtitle_frame.sort('start')



        return data