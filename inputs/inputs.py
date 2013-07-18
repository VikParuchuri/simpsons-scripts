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

    def read_input(self, directory, has_header=True):
        """
        directory is a path to a directory with multiple csv files
        """
    def read_input(self, directory, has_header=True):
        """
        directory is a path to a directory with multiple csv files
        """

        datafiles = [ f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory,f)) and (f.endswith(".srt") or f.endswith(".sub"))]
        all_sub_data = []
        for infile in datafiles:
            stream = open(os.path.join(directory, infile))
            data=stream.read()
            row_data = []
            for (i, row) in enumerate(data):
                if i==0:
                    row_data.append(["Start","End","Line","Label"])
                else:
                    row_split = row.split("}")
                    start = row_split[0].replace("{","")
                    end = row_split[1].replace("{","")
                    line = row_split[2].split("{")[0]
                    if len(row_split[2].split("{"))>1:
                        label = row_split[2].split("{")[1].replace("}","")
                    else:
                        label = ""
                    row_data.append([start,end,line,label])
            all_sub_data.append(row_data)
        sub_data = list(chain.from_iterable(all_sub_data))
        self.data = sub_data