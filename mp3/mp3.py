import numpy as np
from utils import *

net = BBN()
print(net(['summer','north','seabass','dark','thin']))
print(net(['salmon','south','light']))
print(net(['light','south','seabass']))
print(net(['south','light']))
print(net(['salmon','|','south','light']))
print(net(['seabass','|','south','light']))