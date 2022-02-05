import idx2numpy
import numpy as np
import sys 
# Load the training data
train_images = idx2numpy.convert_from_file('train-images.idx3-ubyte')
train_labels = idx2numpy.convert_from_file('train-labels.idx1-ubyte')
test_images = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
test_labels = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')

np.set_printoptions(threshold=sys.maxsize,linewidth=sys.maxsize)




print(train_images[6])

# conv_single_step
# Convolutional Layer 1.
mytrain_images = train_images[0:5000]
print(mytrain_images.shape)
class conv:
    def __init__(self,outputChannels,filterSize,stride,padding,inputChannels):
        self.outputChannels = outputChannels
        self.filterSize = filterSize
        self.stride = stride
        self.padding = padding
        self.inputChannels = inputChannels
        self.weights = np.random.rand(
            outputChannels, filterSize, filterSize, inputChannels)/(filterSize*filterSize*inputChannels)
        self.bias = np.random.randn(outputChannels)
        self.output = None
        self.input = None

    def pad(self,input):
        if self.padding == 0:
            return input
        else:
            pad_input = np.pad(input,((0,0),(self.padding,self.padding),(self.padding,self.padding),(0,0)),'constant',constant_values=0)
            return pad_input

   
    def forward(self,input):
        input = self.pad(input)
        self.input = input
        inputShape = input.shape
        #print("FWD inputshape " , inputShape)
        outoutShape = (inputShape[0],inputShape[1]-self.filterSize+1,inputShape[2]-self.filterSize+1,self.outputChannels)
        self.output = np.zeros(outoutShape)
        #print("FWD outputshape " , outoutShape)
        for i in range(outoutShape[0]):
            for j in range(outoutShape[1]):
                for k in range(outoutShape[2]):
                    for l in range(outoutShape[3]):
                        self.output[i,j,k,l] = np.sum(input[i,j*self.stride:j*self.stride+self.filterSize,k*self.stride:k*self.stride+self.filterSize,:] * self.weights[l,:,:,:]) + self.bias[l]
        return self.output

    def backward(self,grad):
        gradShape = grad.shape
        #print("BWD gradshape " , gradShape)
        grad_input = np.zeros(self.input.shape)
        for i in range(gradShape[0]):
            for j in range(gradShape[1]):
                for k in range(gradShape[2]):
                    for l in range(gradShape[3]):
                        grad_input[i,j*self.stride:j*self.stride+self.filterSize,k*self.stride:k*self.stride+self.filterSize,:] += grad[i,j,k,l] * self.weights[l,:,:,:]
        return grad_input

    


class Relu:
    
    def forward(self,input):
        self.input = input
        output = np.maximum(0,input)
        return output

    def backward(self,prev):
        output = self.input
        output[output>0] = 1
        output[output<=0] = 0
        return output * prev
        
        

class Maxpool:
    def __init__(self,filterSize,stride):
        self.filterSize = filterSize
        self.stride = stride
        self.output = None
        self.input = None
    
    def forward(self,input):
        self.input = input
        inputShape = input.shape
        outputShape = (inputShape[0],int((inputShape[1]-self.filterSize)/self.stride)+1,int((inputShape[2]-self.filterSize)/self.stride)+1,inputShape[3])
        self.output = np.zeros(outputShape)
        for i in range(inputShape[0]):
            for j in range(inputShape[1]-self.filterSize+1):
                for k in range(inputShape[2]-self.filterSize+1):
                    for l in range(inputShape[3]):
                        self.output[i,int(j/self.stride),int(k/self.stride),l] = np.max(input[i,j:j+self.filterSize,k:k+self.filterSize,l])
        return self.output
    
    

        
        

    def backward(self,grad):
        inputShape = self.input.shape
        gradShape = grad.shape
        output = np.zeros(inputShape)
        outShape = self.output.shape
        for i in range(outShape[0]):
            for j in range(outShape[1]):
                for k in range(outShape[2]):
                    for l in range(outShape[3]):
                        print("i=",i,"j=",j,"k=",k,"l=",l)
                        

        return output

        




   
    


class Flatten:
    def __init__(self):
        self.output = None
        self.input = None
    
    def forward(self,input):
        self.input = input
        inputShape = input.shape
        self.output = np.reshape(input,(inputShape[0],inputShape[1]*inputShape[2]*inputShape[3]))
        return self.output

    def backward(self,grad):
        self.grad = grad
        self.gradInput = np.reshape(grad,self.input.shape)
        return self.gradInput


# def fullyconnected(outputChannels,input):
#     inputShape = input.shape
#     weights = np.random.randn(outputChannels,inputShape[0])
#     bias = np.random.randn(outputChannels)
#     output = np.zeros((outputChannels,inputShape[0]))
#     for i in range(outputChannels):
#         for j in range(inputShape[0]):
#             output[i,j] = np.dot(weights[i,:],input[:,j]) + bias[i]
#     return output

class FullyConnected:
    def __init__(self,outputChannels,inputChannels):
        self.outputChannels = outputChannels
        self.inputChannels = inputChannels
        self.weights = np.random.randn(outputChannels,inputChannels)/inputChannels
        self.bias = np.zeros(outputChannels)
        self.output = None
        self.input = None

    def forward(self,input):
        self.input = input.T
        inputShape = input.shape
        output = np.zeros((inputShape[0],self.outputChannels))
        for i in range(inputShape[0]):
            for j in range(self.outputChannels):
                output[i,j] = np.dot(self.weights[j,:],input[i,:]) + self.bias[j]
        self.output = output
        return self.output

    def backward(self,delta):
        deltaShape = delta.shape
       # print("DeltaShape ",deltaShape)
       # print("InputShape ",self.input.shape)
        dWeights = np.zeros((self.outputChannels,self.inputChannels))
        dBias = np.zeros(self.outputChannels)
        dInput = np.zeros(deltaShape)
        for i in range(deltaShape[0]):
            for j in range(self.outputChannels):
                dBias[j] += delta[i,j]
        
        dWeights = np.dot(self.input,delta)
        dWeights = dWeights.T
        dInput = np.dot(delta,self.weights)
        #print("Weights", self.weights.shape )
        return dInput, dWeights, dBias
    
    def updateParams(self,dw,db,learningRate):
        self.weights = self.weights - learningRate*dw
        self.bias = self.bias - learningRate*db
        





class Softmax:
    def __init__(self):
        self.output = None
        self.input = None
        self.delta = None
    
    def forward(self,input):
        self.input = input
        inputShape = input.shape
        output = np.zeros((inputShape[0],inputShape[1]))
        for i in range(inputShape[0]):
            for j in range(inputShape[1]):
                output[i,j] = np.exp(input[i,j])/np.sum(np.exp(input[i,:]))
        self.output = output
        return self.output

    def backward(self,grad):
        return (self.output - grad)/self.output.shape[0]

  
def crossEntropy(output,target):
    outputShape = output.shape
    loss = 0
    for i in range(outputShape[0]):
        for j in range(outputShape[1]):
            loss += -target[i,j]*np.log(output[i,j])
    return loss/outputShape[0]



my_yreal = train_labels[0:35]    ## No of Labels in Training Set

print("Yreal ", my_yreal.shape)
Y_label = np.zeros((my_yreal.size, my_yreal.max()+1))
Y_label[np.arange(my_yreal.size), my_yreal.ravel()] = 1


testconv = conv(6,5,1,2,1)
mytrain_images =  np.reshape(mytrain_images,(5000,28,28,1))
#mytrain_images = testconv.pad(mytrain_images)
print(mytrain_images.shape)
out = testconv.forward(mytrain_images[:35]) # No of Images in Training Set
print("Shape after conv 6 5 1 2",out.shape)
relu = Relu()
out = relu.forward(out)
maxpool = Maxpool(2,2)
out = maxpool.forward(out)
print("Shape after maxpool 2 2 ", out.shape)
conv2 = conv(12,5,1,0,6)
out = conv2.forward(out)
print("Shape after conv 12 5 1 0", out.shape)
out = relu.forward(out)
out = maxpool.forward(out)
print("Shape after maxpool 2 2", out.shape)
conv3 = conv(100,5,1,0,12)
out = conv3.forward(out)
print("Shape after conv 100 5 1 0", out.shape)
out = relu.forward(out)
flat = Flatten()
out = flat.forward(out)
print("Shape after Flatten ", out.shape)
FullyConnected = FullyConnected(10,100)
out = FullyConnected.forward(out)
print("Full connected output shape ", out.shape)
soft = Softmax()
out = soft.forward(out)
print(out.shape)

## backprop

softgrad = soft.backward(Y_label)
delta, dw, db = FullyConnected.backward(softgrad)
print(delta.shape)
FullyConnected.updateParams(dw,db,0.01)
flat_grad = flat.backward(delta)
print(flat_grad.shape)

print("Cross entropy loss = ",crossEntropy(out,Y_label))




