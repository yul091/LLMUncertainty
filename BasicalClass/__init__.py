from .BasicModule import BasicModule
from .CodeSummary import CodeSummary_Module
from .CodeCompletion import CodeCompletion_Module
from .common_function import *


"""
    This dirotory contains different Module and some common use API
    a Module is a object contains the model, training data, testing data etc.
    module list:
"""

MODULE_LIST = [
    CodeSummary_Module,
    CodeCompletion_Module,
]
