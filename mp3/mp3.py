import numpy as np
from utils import *

net = BBN()
print(net.P(['salmon','|','south','light']))
print(net.P(['seabass','|','south','light']))