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
    'Tertiary': ['',
      'Willy',
      'Congressman',
      'Ham',
      'Mr. Burns',
      'Make-Up Man',
      'Alligator',
      'Sherri',
      'Frog 1',
      'Frog 2',
      'Auctioneer',
      'Theme',
      "Lenny'S Voice",
      'Bart &Milhouse',
      'Pele',
      'Pilot One',
      'Fisherman 2',
      'Elves',
      'Fisherman 1',
      'Woman',
      'Mrs. Prince',
      'Frog 3',
      'Hutz',
      'Mrs. Krebappel',
      'Lizzie',
      'Mendoza',
      ',',
      "Homer'S Voice",
      'Mcclure',
      'Fan',
      'Snake',
      'Silverman',
      'Ms.K',
      'Prostitute',
      'Scary Devil',
      'Kids',
      'Bart & Milhouse',
      'H',
      'Tv Presenter',
      'Mcbain',
      'Billy',
      'Inspector & Woman',
      'Mother',
      'Mr. Largo',
      'Kearney',
      'Rev. Lovejoy',
      'Louie',
      'Executioner',
      'Grim Reaper',
      'Crowd',
      'Meek Voice',
      'Cohen',
      'Gallagher',
      'Burns Robot',
      'Mccartney',
      'Jasper',
      'T-Shirt Vendor',
      'Tv Voiceover',
      'Kid',
      'Gummy Joe',
      'George',
      'Hans Moleman',
      'Mrs. Glick',
      'Hibbert',
      'Men',
      'Family & Apu',
      'Announcer',
      'Guitarist',
      'Dr Hibbert',
      'Scully',
      'Ringo',
      'Bart & Barney',
      'Marvin',
      'Dr. Joyce Brothers',
      'Email',
      'Arsonist',
      'Cameraman',
      'Terri',
      'Ramone #4',
      'Clerk',
      'Bart & Lisa',
      'Worker',
      'Ramone #1',
      'Ramone #2',
      'Ramone #3',
      'Mrs. Lovejoy',
      'Frink',
      'Game',
      'Mulder',
      'Technician',
      'Librarian',
      'Uter',
      'Class',
      'Robber',
      'Meyer',
      'Boy On Tv',
      'Boy',
      'Scotsman',
      'Mr. Rogers',
      'Poochie',
      'Adviser',
      "Burns' Grandfather",
      'Customers',
      'Sideshow Mel',
      'S',
      'Maude',
      'Bailey',
      'Spanish Bee',
      'Inspector',
      'Shutton',
      'Maggie',
      'Dr. Nick',
      'Searing Pain!?',
      'Clinton',
      'Kang',
      'Super Friends',
      'Alien',
      'Roy',
      'Comedian',
      'Radio Dj',
      'Lawyer',
      'Todd',
      'Salesman',
      'Nixon',
      'Pilot Two',
      'Nooooo!!!',
      'Doctor',
      'Man 1',
      'Vulture',
      'Man 2',
      'Robots',
      'Dr. Hibbert',
      'Lou',
      'Tv Repair Man',
      'Lovejoy',
      'Tom Savini',
      'Dr Nick',
      'Conductor',
      'Mexican Commentator',
      'Wendell',
      'Selma',
      'Cashier',
      '.',
      'Moleman',
      'Tv Announcer',
      'Hans',
      'Agnes',
      'Nimoy',
      'Tannoy',
      'Bumblebee Guy',
      'Singer',
      'Carl',
      'Ralph',
      'Motel Clerk',
      'Don Homer',
      'Flanders',
      'Voice',
      'Agnes Skinner',
      'Secretary',
      'Bellamy',
      'Shop Assistant',
      'Scratchy',
      'Darwin',
      'Family',
      'Demon',
      'General',
      'Writers',
      'Guests',
      'Harrison',
      'C.E.O.',
      'Speaker',
      'Edna Krebappel',
      'Blackbeard',
      'Abe Simpson',
      'Hitler',
      'Reporter',
      'Lionel Hutz',
      'Jimbo',
      'Abe',
      'Floor',
      'Tv',
      'God',
      'Father',
      'Oakley',
      'Bush',
      'Drummer',
      'Teenager',
      'Arnie',
      'Jingle',
      'Itchy',
      'Pianist',
      'Eagle',
      'Captain Mccallister',
      'Narrator',
      'Tester',
      'Advisor 3',
      'Advisor 2',
      'Santas',
      'Guard',
      'Bear',
      'Toucan',
      'Elf #2',
      'Elf #1',
      'Teacher',
      'Photographer',
      'Tattoo Guy',
      'Scott Christian',
      'Employees',
      'Doug',
      'Arnold',
      'Report Card',
      'Guards',
      'Hot Dog Vendor',
      'Cletus',
      'Dr. Wolfe',
      'Miss Hoover',
      'Abraham',
      'D-O-G.',
      'Strikers',
      'Gates',
      'All',
      'Everyone',
      'Carter',
      'Reverend Lovejoy',
      'Alf',
      'Lewis',
      'Executive',
      'Hobo',
      'Indian Man',
      'Player',
      'Manager',
      'Homer Slaves',
      'Girl',
      'Sarcastic Clerk',
      'Smithers Robot',
      "Homer'S Brain"],
     'Marge': ['Marge'],
     'Homer': ['Homer'],
     'Burns': ['Burns'],
     'Secondary': ['Kent', 'Moe', 'Ned', 'Smithers', 'Apu', 'Skinner', 'Milhouse'],
     'Lisa': ['Lisa'],
     'Bart': ['Bart'],
}

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

        vec = Vectorizer()

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

        vec.fit(speaker_list, speaker_codes, 200,min_features=1)
        features = vec.batch_get_features(speaker_list)

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
        pyplot.legend(loc=3)
        pyplot.savefig('clusters.png')

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
