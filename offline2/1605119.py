from os import system
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


def stateProb(transitionMatrix):
    arr = np.array(transitionMatrix)
    S,U = eig(arr.T)
    stateProb = np.array(U[:, np.where(np.abs(S - 1.) < 1e-8)[0][0]].flat)
    stateProb = stateProb/np.sum(stateProb)
    return stateProb

def EmissionProb(means, variences):
    emissionProb = []
    Std_dev = []
    for i in range(states):
        Std_dev.append(math.sqrt(variences[i]))
    for i in range(states):
        emissionProb.append(list(norm.pdf(data, loc=means[i], scale=Std_dev[i])))
    print(np.shape(emissionProb))
    return emissionProb



def Viterbi(Obs, StateSpace, prob, Tm, Em): 
 # To hold p. of each state given each observation.
    Tm = np.array(Tm)
    Em = np.array(Em)
    V = np.zeros((len(Obs), len(StateSpace)))
    # To hold backtrack of each state given each observation.
    Backtrack = np.zeros((len(Obs), len(StateSpace)), dtype=int)
    # Initialize first column of V.
    for i in range(len(StateSpace)):
        V[0][i] = prob[i] * Em[i][0]
        Backtrack[0][i] = 0
    # Fill in the rest of the table.
    for t in range(1, len(Obs)):
        for i in range(len(StateSpace)):
            max_prob = 0
            max_index = 0
            for j in range(len(StateSpace)):
                if V[t - 1][j] * Tm[j][i] > max_prob:
                    max_prob = V[t - 1][j] * Tm[j][i]
                    max_index = j
            V[t][i] = max_prob * Em[i][t]
            Backtrack[t][i] = max_index
        V[t] = V[t] / np.sum(V[t])
    # Find the maximum probability.
    max_prob = 0
    max_index = 0
    for i in range(len(StateSpace)):
        if V[len(Obs) - 1][i] > max_prob:
            max_prob = V[len(Obs) - 1][i]
            max_index = i
    # Follow the backtrack to find the states.
    opt = [max_index]
    for t in range(len(Obs) - 1, 0, -1):
        opt.insert(0, Backtrack[t][opt[0]])
    return opt



np.set_printoptions(threshold=100)

data = loadData() 
loadparams()
# print(transitionMatrix)
# print(means)
# print(variences)
# print(stateProb())
# print(EmissionProb())
startP= stateProb(transitionMatrix)
emissionP = EmissionProb(means, variences)
print(Viterbi(data,[0,1],startP,transitionMatrix,emissionP))




def backward(states, data, emissionP, transitionMatrix):
    beta = np.zeros((len(data), states))
    for i in range(states):
        beta[len(data) - 1][i] = 1
    for t in range(len(data) - 1, 0, -1):
        for i in range(states):
            sum = 0
            for j in range(states):
                sum += beta[t][j] * transitionMatrix[i][j] * emissionP[j][t]
            beta[t - 1][i] = sum
        beta[t - 1] = beta[t - 1] / np.sum(beta[t - 1])
    return beta




def forward(states, data, emissionP, transitionMatrix, startP):
    alpha = np.zeros((len(data), states))
    for i in range(states):
        alpha[0][i] = startP[i] * emissionP[i][0]
    for t in range(1, len(data)):
        for i in range(states):
            sum = 0
            for j in range(states):
                sum += alpha[t - 1][j] * transitionMatrix[j][i]
            alpha[t][i] = sum * emissionP[i][t]
        alpha[t] = alpha[t] / np.sum(alpha[t])

    return alpha


#print(forward())
# print(backward())

Alpha = forward(states, data, emissionP, transitionMatrix, startP)
Beta = backward(states, data, emissionP, transitionMatrix)


forward_sink = np.sum(Alpha[len(data) - 1])

def CalculatePi_star(states, Alpha, Beta, data,forward_sink):
    Pi_star = np.zeros((len(data), states))
    for i in range(len(data)):
        for j in range(states):
            Pi_star[i][j] = Alpha[i][j] * Beta[i][j] / forward_sink
        Pi_star[i] = Pi_star[i] / np.sum(Pi_star[i])
    return Pi_star


def CalculatePiStarStar(states,data,Alpha,Beta,transitionMatrix,emissionP,forward_sink):
    Pi_star_star = np.zeros((len(data) -1, states, states))
    for i in range(len(data) - 1):
        for j in range(states):
            for k in range(states):
               # Pi_star_star[i][j][k] = Alpha[i][j] * transitionMatrix[j][k] * emissionP[k][i + 1] * Beta[i + 1][k] / forward_sink
                Pi_star_star[i][j][k] = Alpha[i][k] * transitionMatrix[k][j] *  emissionP[j][i + 1] * Beta[i + 1][j] / forward_sink
        Pi_star_star[i] = Pi_star_star[i] / np.sum(Pi_star_star[i])
    return Pi_star_star



pistar = CalculatePi_star(states, Alpha, Beta, data,forward_sink)
pi2star = CalculatePiStarStar(states,data,Alpha,Beta,transitionMatrix,emissionP,forward_sink)

def updateTransitionMatrix(pi2starM):
    global transitionMatrix
    transitionMatrix = transitionMatrix
    for i in range(states):
        for j in range(states):
            transitionMatrix[i][j] = np.sum(pi2starM[:,i,j])
    
    for i in range(states):
        transitionMatrix[i] = transitionMatrix[i] / np.sum(transitionMatrix[i])

    print(transitionMatrix)
    print(np.shape(transitionMatrix))
    return transitionMatrix


def updateMeans(pistarM):
    global means
    means = means
    print(means)
    for i in range(states):
        means[i] = np.sum(pistarM[:,i] * data) / np.sum(pistarM[:,i])
    print(means)
    return means

def updateVariences(pistarM):
    global variences
    variences = variences
    print(variences)
    for i in range(states):
        variences[i] = np.sum(pistarM[:,i] * (data - means[i]) ** 2) / np.sum(pistarM[:,i])
    print(variences)
    return variences




def baum_welch(transitionMatrix, means, variences, data, states,emissionP,startP):
    for i in range(5):
        print("iteration: ", i)
        print("transitionMatrix: ", transitionMatrix)
        print("means: ", means)
        print("variences: ", variences)
        print("startP: ", startP)
        print("\n")
        Alpha = forward(states, data, emissionP, transitionMatrix, startP)
        Beta = backward(states, data, emissionP, transitionMatrix)
        forward_sink = np.sum(Alpha[len(data) - 1])
        Pi_star = CalculatePi_star(states, Alpha, Beta, data,forward_sink)
        pi2star = CalculatePiStarStar(states,data,Alpha,Beta,transitionMatrix,emissionP,forward_sink)
        transitionMatrix = updateTransitionMatrix(pi2star)
        startP = stateProb(transitionMatrix)
        means = updateMeans(Pi_star)
        variences = updateVariences(Pi_star)
        emissionP = EmissionProb(means, variences)


baum_welch(transitionMatrix, means, variences, data, states,emissionP,startP)
#print(pi2star)


