from matplotlib import pyplot as plt
from tqdm import trange
from tensorflow import tf
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from matplotlib.pyplot import plt 
from torchvision import transforms, models
from torchvision.datasets.cifar import CIFAR10
import numpy

from torchsummary import summary

torch.manual_seed(0)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')