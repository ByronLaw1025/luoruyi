import pandas as pd  
import numpy as np
from tqdm import tqdm  
from collections import defaultdict  
import os, math, warnings, math, pickle
from tqdm import tqdm
import faiss
import collections
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from deepctr.feature_column import SparseFeat, VarLenSparseFeat
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from deepmatch.models import *
from deepmatch.utils import sampledsoftmaxloss
warnings.filterwarnings('ignore')