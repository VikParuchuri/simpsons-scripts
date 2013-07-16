from pandas import DataFrame
import numpy as np
from percept.utils.models import FieldModel

from percept.fields.base import Dict
from percept.conf.base import settings
from percept.utils.models import RegistryCategories, get_namespace
from percept.utils.input import DataFormats
from percept.tests.framework import JSONFormatTester
from percept.datahandlers.formatters import BaseFormat, JSONFormat
from inputs.inputs import SimpsonsFormats
import os
import re
import logging
log = logging.getLogger(__name__)

class ScriptFormatter(JSONFormat):
    namespace = get_namespace(__module__)

    def from_script(self,input_data):
        """
        Reads script format input data, but data is already in json, so return.
        """
        return input_data


