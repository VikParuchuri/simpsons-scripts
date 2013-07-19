from __future__ import division
import csv
from percept.conf.base import settings
from percept.utils.input import DataFormats
from percept.tests.framework import CSVInputTester
from percept.datahandlers.inputs import BaseInput
from percept.utils.models import get_namespace
import os
from itertools import chain
import logging
import json
import re
log = logging.getLogger(__name__)

class SimpsonsFormats(DataFormats):
    script = "script"
    subtitle = "subtitle"

class ScriptInput(BaseInput):
    """
    Extends baseinput to read simpsons scripts
    """
    input_format = SimpsonsFormats.script
    help_text = "Reformat simpsons script data."
    namespace = get_namespace(__module__)

    def read_input(self, filename, has_header=True):
        """
        directory is a path to a directory with multiple csv files
        """

        filestream = open(filename)
        self.data = json.load(filestream)

class SubtitleInput(BaseInput):
    """
    Extends baseinput to read simpsons scripts
    """
    input_format = SimpsonsFormats.subtitle
    help_text = "Reformat simpsons script data."
    namespace = get_namespace(__module__)


    def get_episode_metadata(self, name):
        episode_code = re.search("\[\d+\.\d+\]", name).group(0).replace("[","").replace("]","")
        season, episode = episode_code.split(".")
        season = int(season)
        episode = int(episode)
        return season, episode

    def read_input(self, directory, has_header=True):
        """
        directory is a path to a directory with multiple csv files
        """

        sub_datafiles = [ f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory,f)) and f.endswith(".sub")]
        all_sub_data = []
        for infile in sub_datafiles:
            stream = open(os.path.join(directory, infile))
            season,episode = self.get_episode_metadata(infile)
            data=stream.read()
            row_data = []
            for (i, row) in enumerate(data.split("\n")):
                row = row.replace('\r','')
                row_split = row.split("}")
                if len(row_split)>2:
                    start = float(row_split[0].replace("{",""))/24
                    end = float(row_split[1].replace("{",""))/24
                    line = row_split[2].split("{")[0]
                    if len(row_split[2].split("{"))>1:
                        label = row_split[2].split("{")[1].replace("}","")
                    else:
                        label = ""
                    row_data.append([start,end,line,label,season,episode])
            all_sub_data.append(row_data)
        srt_datafiles = [ f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory,f)) and f.endswith(".srt")]
        for infile in srt_datafiles:
            stream = open(os.path.join(directory, infile))
            season,episode = self.get_episode_metadata(infile)
            data=stream.read()
            row_data = []
            for (i, row) in enumerate(data.split("\r\n\r\n")):
                row_split = row.split("\r\n")
                if len(row_split)>3:
                    timing = row_split[1]
                    start = float(timing.split("-->")[0].replace(",",".").split(":")[-1])
                    end = float(timing.split("-->")[1].replace(",",".").split(":")[-1])
                    line = " ".join(row_split[2:])
                    if len(line.split("{"))>1:
                        label = line.split("{")[1].replace("}","")
                        line = line.split("{")[0]
                    else:
                        label = ""
                    row_data.append([start,end,line,label,season,episode])
            all_sub_data.append(row_data)
        sub_data = [["start","end","line","label","season","episode"]] + list(chain.from_iterable(all_sub_data))
        self.data = sub_data