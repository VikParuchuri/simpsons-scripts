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

log = logging.getLogger(__name__)

PUNCTUATION = ["]",".","!","?"]

CHARACTERS = {
    'Tertiary': ['Kent',
                 'Ms.K',
                 'Nelson',
                 'Rev. Lovejoy',
                 'Announcer',
                 'Meyers',
                 'Krusty',
                 'Alien',
                 'Woods',
                 'Doctor',
                 'Man 1',
                 'Dr. Hibbert',
                 'Otto',
                 'Reporter',
                 'Selma',
                 'Brockman',
                 'Carl',
                 'Wiggum',
                 'Flanders',
                 'Comic Book Guy',
                 'Grampa',
                 'Kent Brockman',
                 'Advisor',
                 'Man',
                 'Lenny',
                 'Guards',
                 'Executive'],
     'Marge': ['Marge'],
     'Homer': ['Homer'],
     'Burns': ['Burns'],
     'Secondary': ['Moe', 'Ned', 'Smithers', 'Apu', 'Skinner', 'Milhouse', 'Grimes'],
     'Lisa': ['Lisa'],
     'Bart': ['Bart'],
}

CHARACTER_REPLACEMENT = {
    'H' : "Homer",
    'M' : "Marge",
    "L" : "Lisa",
    "B" : "Bart",
    "G" : "Grampa",
    "Abe" : "Grampa",
    "Principal Skinner" : "Skinner",
    "Ms. K" : "Ms.K",
    "Mrs. Krebappel" : "Ms.K",
    "Mr. Burns" : "Burns",
    "Mcclure": "Troy",
    "Edna Krebappel": "Ms.K",
    "Abraham" : "Grampa",
    "Abe Simpson" : "Grampa",
    "Hutz" : "Lionel Hutz",
}

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

def check_if_character(character):
    all_characters = list(chain.from_iterable([CHARACTERS[k] for k in CHARACTERS]))
    return character in all_characters

def find_replacement(character):
    for k in CHARACTERS:
        if character in CHARACTERS[k]:
            return k
    return "Tertiary"

def cleanup_name(character):
    for k in CHARACTER_REPLACEMENT:
        if k==character:
            character = CHARACTER_REPLACEMENT[k]
    return character

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
        return check_if_character(line.split(":")[0])

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

    args = {'scriptfile' : os.path.abspath(os.path.join(settings.PROJECT_PATH, "data/raw_scripts2.json")), 'do_replace' : True}

    def train(self, data, target, **kwargs):
        """
        Used in the training phase.  Override.
        """
        self.data = data
        self.predict(self.data, **kwargs)

    def predict(self, data, **kwargs):
        """
        Used in the predict phase, after training.  Override
        """

        voice_scripts = list(data['voice_script'])
        scriptfile = kwargs['scriptfile']
        do_replace = kwargs['do_replace']
        json_scripts = json.load(open(scriptfile))
        voice_scripts+=[s['script'] for s in json_scripts]
        script_segments = []
        for script in voice_scripts:
            script = script.replace("\"","")
            lines = script.split("\n")
            segment = []
            for line in lines:
                if line.strip()!="":
                    line = line.encode('ascii','ignore')
                    line_split = line.split(":")
                    if do_replace:
                        line_split[0] = find_replacement(line_split[0].strip())
                    line_split[0] = cleanup_name(line_split[0].strip())
                    segment.append({'speaker' : line_split[0],
                                    'line' : ":".join(line_split[1:]).strip()})
                else:
                    if len(segment)>0:
                        script_segments.append(segment)
                        segment = []
            if len(segment)>0:
                script_segments.append(segment)
        self.voice_lines = script_segments

class ClusterScriptText(Task):
    data = Complex()
    clusters = Complex()
    predictions = Complex()
    clusters = List()
    cl = Complex()
    vec = Complex()
    vec1 = Complex()

    data_format = SimpsonsFormats.dataframe

    category = RegistryCategories.preprocessors
    namespace = get_namespace(__module__)

    help_text = "Cluster simpsons scripts."

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

        from train import Vectorizer, make_df

        self.vec = Vectorizer()

        reformatter = ReformatScriptText()
        args = reformatter.args
        args['do_replace'] = False
        reformatter.train(data, "", **args)

        script_segments = list(chain.from_iterable(reformatter.voice_lines))
        text = [s['line'] for s in script_segments]
        speaker = [s['speaker'] for s in script_segments]
        unique_speakers = list(set(speaker))
        speaker_code_dict = {k:i for (i,k) in enumerate(unique_speakers)}
        speaker_codes = [speaker_code_dict[k] for k in unique_speakers]
        speaker_list = []
        speaker_frame = make_df([text,speaker],["text","speaker"])
        for i in unique_speakers:
            s_text = "\n".join(list(speaker_frame[speaker_frame['speaker']==i]['text']))
            speaker_list.append(s_text)

        self.vec.fit(speaker_list, speaker_codes, 200,min_features=2)
        features = self.vec.batch_get_features(speaker_list)

        cl = KMeans()
        self.predictions = cl.fit_predict(features)
        self.cl = cl

        for i in xrange(0,max(self.predictions)):
            clust = []
            for c in xrange(0,len(speaker_codes)):
                if self.predictions[c]==i:
                    clust.append(unique_speakers[c])
            self.clusters.append(clust)

        pca = PCA(n_components=2, whiten=True).fit(features)
        rf = pca.transform(features)
        labels = cl.labels_
        pyplot.clf()
        centroids = cl.cluster_centers_
        pyplot.cla()
        for i in range(max(labels)):
            ds = rf[np.where(labels==i)]
            pyplot.plot(ds[:,0],ds[:,1],'o', label=self.clusters[i][0])
        pyplot.legend(loc=8)
        pyplot.savefig('clusters.png')

        self.vec1 = Vectorizer()
        speaker_codes = [speaker_code_dict[k] for k in speaker]

        self.vec1.fit(text, speaker_codes, 200,min_features=2)
        features = self.vec1.batch_get_features(text)

        pca = PCA(n_components=2, whiten=True).fit(features)
        rf = pca.transform(features)
        pyplot.clf()
        pyplot.cla()
        for i in range(len(speaker_codes)):
            pyplot.plot(rf[i,0],rf[i,1],'o', label=speaker[i])
        pyplot.savefig('all_speakers.png')


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
