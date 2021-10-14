import math
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from time import time, sleep
import argparse
import pdb
from rdkit import Chem
from datetime import datetime
from heapq import heappush, heappop
from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')
from itertools import count


# Deep learning modules
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
#from CustomTransformer import *
from torch.nn import Transformer
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
import gc
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts