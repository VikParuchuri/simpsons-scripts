from __future__ import division
from percept.tasks.base import Task
from percept.tasks.train import Train
from percept.fields.base import Complex, List, Dict, Float
from inputs.inputs import SimpsonsFormats
from percept.utils.models import RegistryCategories, get_namespace
import logging
import numpy as np
import calendar
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import math
import random
from itertools import chain
from percept.tests.framework import Tester
import os
from percept.conf.base import settings
import re

log = logging.getLogger(__name__)

PUNCTUATION = ["]",".","!","?"]

def make_df(datalist, labels, name_prefix=""):
    df = pd.DataFrame(datalist).T
    if name_prefix!="":
        labels = [name_prefix + "_" + l for l in labels]
    labels = [l.replace(" ", "_").lower() for l in labels]
    df.columns = labels
    df.index = range(df.shape[0])
    return df

class CleanupScriptList(Task):
    data = Complex()

    data_format = SimpsonsFormats.dataframe

    category = RegistryCategories.preprocessors
    namespace = get_namespace(__module__)

    help_text = "Cleanup simpsons scripts."

    def train(self, data, target, **kwargs):
        """
        Used in the training phase.  Override.
        """
        self.data = self.predict(data)

    def predict(self, data, **kwargs):
        """
        Used in the predict phase, after training.  Override
        """

        script_removal_values = [""]
        for r in script_removal_values:
            data = data[data["script"]!=r]
        log.info(data)
        data['episode_name'] = [i.split('\n')[0].strip() for i in data['episode_name']]
        data['episode_code'] = [i.split('/')[-1].split('.html')[0] for i in data['url']]

        data.index = range(data.shape[0])
        return data

class CleanupScriptText(Task):
    data = Complex()

    data_format = SimpsonsFormats.dataframe

    category = RegistryCategories.preprocessors
    namespace = get_namespace(__module__)

    help_text = "Cleanup simpsons scripts."

    def train(self, data, target, **kwargs):
        """
        Used in the training phase.  Override.
        """
        self.data = self.predict(data)

    def predict(self, data, **kwargs):
        """
        Used in the predict phase, after training.  Override
        """

        voice_scripts = []
        for i in xrange(0,data.shape[0]):
            script_lines = data['script'][i].split('\n')
            voice_lines = []
            current_line = ""
            for (i,line) in enumerate(script_lines):
                line = line.strip()
                if line.startswith("[") and line.endswith("]"):
                    continue
                voice_line = re.search('\w+:',line)
                if voice_line is not None:
                    voice_lines.append(current_line)
                    current_line = line
                elif len(line)==0 and len(current_line)>0:
                    voice_lines.append(current_line)
                    current_line = ""
                elif len(current_line)>0:
                    current_line+=line
            voice_scripts.append("\n".join([l for l in voice_lines if len(l)>0 and "{" not in l and "=" not in l]))

        data['voice_script'] = voice_scripts

        return data