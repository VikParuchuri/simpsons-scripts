from __future__ import division
from percept.tasks.base import Task
from percept.tasks.train import Train
from percept.fields.base import Complex, List, Dict, Float
from inputs.inputs import SimpsonsFormats
from percept.utils.models import RegistryCategories, get_namespace
import logging
import pandas as pd
from percept.tests.framework import Tester
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
    "Ms. K:" : "Ms.K",
}

SECONDARY_CHARACTERS = [
    u'Kent',
    u'Lenny',
    u'Quimby',
    u'Troy',
    u'Martin',
    u'Otto',
    u'Selma',
    u'Patty',
    u'Ms.K',
    u'Hutz',
    u'Nelson',
    u'Barney',
    u'Grampa',
    u'Maggie',
    u'Apu',
    u'Krusty',
    u'Moe',
]

CHARACTERS = [
    u'Lisa',
    u'Bart',
    u'Marge',
    u'Homer',
    u'Skinner',
    u'Burns',
    u'Ned',
    u'Wiggum',
    u'Milhouse',
    u'Smithers',
    u'Sideshow Bob',
    u'Secondary'
]

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
                for k in SECONDARY_CHARACTERS:
                    line = re.sub(k+":","Secondary:",line)
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
                    line = line.encode('ascii','ignore')
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
            script_lines = data['script'][i].split('\r \r')
            lines = []
            current_line = ""
            for line in script_lines:
                line = line.replace("-","")
                line = line.replace("\n", "")
                line = line.replace("\\","")
                line = line.strip()
                line = line.encode('ascii','ignore')
                current_line= "{0} {1}".format(current_line,line)
                if current_line[-1] in PUNCTUATION:
                    lines.append(current_line)
                    current_line = ""
            lines = ("\n".join(lines))
            lines = re.sub("\[.+\]","",lines)
            lines = re.sub("[\r\t]","",lines)
            lines = [l.strip() for l in lines.split("\n") if len(l)>10]
            text = "\n".join(lines)
            voice_scripts.append(text)
        data['voice_script'] = voice_scripts
        return data
