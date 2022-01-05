import numpy as np
import math
from scipy.linalg import eig
from scipy.stats import norm
from scipy.stats.morestats import Std_dev

variences = []
means = []
transitionMatrix = []
states = 0


def loadData():
    data = []
    with open("data.txt", "r") as f:
        for line in f:
            data.append(float(line.strip()))
    return data


def loadparams():
    c = 0
    global states
    global variences
    global transitionMatrix
    global means
    with open("parameters.txt.txt", "r") as f:
        for line in f:
            if c == 0:
                states = int(line.strip())
            elif c >= 1 and c <= states:
                transitionMatrix.append(list(map(float, line.strip().split())))
            elif c==states+1:
                means = (list(map(float, line.strip().split())))
            elif c==states+2:   
                variences = (list(map(float, line.strip().split())))
               
            c += 1


def stateProb():
    arr = np.array(transitionMatrix)
    S,U = eig(arr.T)
    stateProb = np.array(U[:, np.where(np.abs(S - 1.) < 1e-8)[0][0]].flat)
    stateProb = stateProb/np.sum(stateProb)
    return stateProb

def EmissionProb():
    emissionProb = []
    Std_dev = []
    for i in range(states):
        Std_dev.append(math.sqrt(variences[i]))

    for i in range(states):
        emissionProb.append(norm.pdf(means[i], loc=means[i], scale=Std_dev[i]))
    return emissionProb
   
    

data = loadData() 
loadparams()
print(transitionMatrix)
print(means)
print(variences)
print(stateProb())
print(EmissionProb())

