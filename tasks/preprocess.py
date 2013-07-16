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

CHARACTER_REPLACEMENT = {
    'H:' : "Homer:",
    'M:' : "Marge:",
    "L:" : "Lisa:",
    "B:" : "Bart:",
    "G:" : "Grampa:",
    "Abe:" : "Grampa:",
    "Principal Skinner:" : "Skinner:",
    "Ms. K:" : "Ms.K"
}

CHARACTERS = [
    u'Burns',
    u'Lenny',
    u'Skinner',
    u'Martin',
    u'Quimby',
    u'Barney',
    u'Apu',
    u'Kent',
    u'Grampa',
    u'Maggie',
    u'Troy',
    u'Ned',
    u'Otto',
    u'Patty',
    u'Lisa',
    u'Selma',
    u'Krusty',
    u'Sideshow Bob',
    u'Bart',
    u'Smithers',
    u'Marge',
    u'Wiggum',
    u'Moe',
    u'Homer',
    u'Hutz',
    u'Ms.K',
    u'Nelson',
    u'Milhouse'
]

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

    def check_for_line_split(self, line):
        return line.split(":")[0] in CHARACTERS

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
                current_line = current_line.strip()
                line = line.strip()
                for k in CHARACTER_REPLACEMENT:
                    line = re.sub(k,CHARACTER_REPLACEMENT[k],line)
                if line.startswith("[") and line.endswith("]"):
                    continue
                if line.startswith("-"):
                    continue
                voice_line = re.search('\w+:',line)
                if voice_line is not None:
                    if self.check_for_line_split(current_line):
                        voice_lines.append(current_line)
                    current_line = line
                elif (len(line)==0 or line.startswith("-")) and len(current_line)>0:
                    if self.check_for_line_split(current_line):
                        voice_lines.append(current_line)
                    current_line = ""
                    voice_lines.append(" ")
                elif len(current_line)>0:
                    current_line+=" " + line
            script_text = "\n".join([l for l in voice_lines if len(l)>0 and "{" not in l and "=" not in l])
            script_text = re.sub("\[.+\]","",script_text)
            voice_scripts.append(script_text.strip())

        data['voice_script'] = voice_scripts

        return data

class ReformatScriptText(Task):
    data = Complex()
    voice_lines = Complex()

    data_format = SimpsonsFormats.dataframe

    category = RegistryCategories.preprocessors
    namespace = get_namespace(__module__)

    help_text = "Cleanup simpsons scripts."

    def train(self, data, target, **kwargs):
        """
        Used in the training phase.  Override.
        """
        self.data = data
        self.predict(self.data)

    def predict(self, data, **kwargs):
        """
        Used in the predict phase, after training.  Override
        """

        script_segments = []
        for script in data['voice_script']:
            lines = script.split("\n")
            segment = []
            for line in lines:
                if line.strip()!="":
                    line_split = line.split(":")
                    segment.append({'speaker' : line_split[0].strip(),
                                    'line' : ":".join(line_split[1:]).strip()})
                else:
                    if len(segment)>0:
                        script_segments.append(segment)
                        segment = []
        self.voice_lines = script_segments

class CleanupTranscriptList(CleanupScriptList):
    help_text = "Cleanup simpsons transcripts."

    def predict(self, data, **kwargs):
        """
        Used in the predict phase, after training.  Override
        """

        script_removal_values = [""]
        for r in script_removal_values:
            data = data[data["script"]!=r]
        episode_names = []
        for i in data['url']:
            match = re.search("episode=\w+",i)
            if match is None:
                episode_names.append("s00e00")
                continue
            episode_names.append(match.group(0).split("=")[1])

        data['episode_name'] = episode_names
        data['season'] = [int(i.split("e")[0].replace("s","")) for i in episode_names]
        data['episode_number'] = [int(i.split("e")[1]) for i in episode_names]

        data.index = range(data.shape[0])
        return data

class CleanupTranscriptText(CleanupScriptText):

    help_text = "Cleanup simpsons transcripts."

    def predict(self, data, **kwargs):
        """
        Used in the predict phase, after training.  Override
        """
        voice_scripts = []
        for i in xrange(0,data.shape[0]):
            script_lines = data['script'][i].split('\n')
            for line in script_lines:
                line = re.sub("-","",line)
                line = line.strip()
        return data

