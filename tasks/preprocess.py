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

log = logging.getLogger(__name__)

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

class CleanupScriptTexgt(Task):
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

        for i in xrange(0,data.shape[0]):
            pass