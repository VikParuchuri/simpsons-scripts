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