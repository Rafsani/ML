from pyexpat import model
import idx2numpy
import numpy as np
import sys 
# Load the training data
train_images = idx2numpy.convert_from_file('train-images.idx3-ubyte')
train_labels = idx2numpy.convert_from_file('train-labels.idx1-ubyte')
test_images = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
test_labels = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')

np.set_printoptions(threshold=sys.maxsize,linewidth=sys.maxsize)




#print(train_images[6])

# conv_single_step
# Convolutional Layer 1.

#print(mytrain_images.shape)
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

    def unpad(self,input):
        if self.padding == 0:
            return input
        else:
            unpad_input = input[:,self.padding:-self.padding,self.padding:-self.padding,:]
            return unpad_input

   
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

    def backward(self,delta):
        deltaShape = delta.shape
        #print("BWD deltaShape " , deltaShape)
        dW = np.zeros(self.weights.shape)
        db = np.zeros(self.bias.shape)
        dInput = np.zeros(self.input.shape)
        for i in range(deltaShape[0]):
            for j in range(deltaShape[1]):
                for k in range(deltaShape[2]):
                    for l in range(deltaShape[3]):
                        dW[l,:,:,:] += delta[i,j,k,l] * self.input[i,j*self.stride:j*self.stride+self.filterSize,k*self.stride:k*self.stride+self.filterSize,:]
                        db[l] += delta[i,j,k,l]
                        dInput[i,j*self.stride:j*self.stride+self.filterSize,k*self.stride:k*self.stride+self.filterSize,:] += delta[i,j,k,l] * self.weights[l,:,:,:]
        dInput = self.unpad(dInput)
        self.update_params(dW,db,0.001)
        return dInput

    def update_params(self,dw,db,learningRate):
        self.weights -= learningRate * dw
        self.bias -= learningRate * db
    


class Relu:
    
    def forward(self,input):
        self.input = input
        output = np.maximum(0,input)
        return output

    def backward(self,prev):
        delta = np.zeros(prev.shape)
        delta[self.input>0] = prev[self.input>0]
        return delta
        
        
        

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
                        output[i,j*self.stride:j*self.stride+self.filterSize,k*self.stride:k*self.stride+self.filterSize,l] = grad[i,j,k,l] * (self.output[i,j,k,l] == self.input[i,j*self.stride:j*self.stride+self.filterSize,k*self.stride:k*self.stride+self.filterSize,l])
                        

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
        self.updateParams(dWeights,dBias,0.001)
        return dInput
    
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
    Ytest = np.zeros(output.shape)
    Ytest[np.arange(target.size), target.ravel()] = 1
    outputShape = output.shape
    loss = 0
    for i in range(outputShape[0]):
        for j in range(outputShape[1]):
            loss += -Ytest[i,j]*np.log(output[i,j])
    return loss/outputShape[0]



my_yreal = train_labels[0:35]    ## No of Labels in Training Set

print("Yreal ", my_yreal.shape)
Y_label = np.zeros((my_yreal.size, my_yreal.max()+1))
Y_label[np.arange(my_yreal.size), my_yreal.ravel()] = 1


# testconv = conv(6,5,1,2,1)
# mytrain_images =  np.reshape(mytrain_images,(5000,28,28,1))
# #mytrain_images = testconv.pad(mytrain_images)
# print(mytrain_images.shape)
# out = testconv.forward(mytrain_images[:35]) # No of Images in Training Set
# print("Shape after conv 6 5 1 2",out.shape)
# relu1 = Relu()
# out = relu1.forward(out)
# maxpool1 = Maxpool(2,2)
# out = maxpool1.forward(out)
# print("Shape after maxpool 2 2 ", out.shape)
# conv2 = conv(12,5,1,0,6)
# out = conv2.forward(out)
# print("Shape after conv 12 5 1 0", out.shape)
# relu2 = Relu()
# out = relu2.forward(out)
# maxpool2 = Maxpool(2,2)
# out = maxpool2.forward(out)
# print("Shape after maxpool 2 2", out.shape)
# conv3 = conv(100,5,1,0,12)
# out = conv3.forward(out)
# print("Shape after conv 100 5 1 0", out.shape)
# relu3 = Relu()
# out = relu3.forward(out)
# flat = Flatten()
# out = flat.forward(out)
# print("Shape after Flatten ", out.shape)
# FullyConnected = FullyConnected(10,100)
# out = FullyConnected.forward(out)
# print("Full connected output shape ", out.shape)
# soft = Softmax()
# out = soft.forward(out)
# print(out.shape)

# ## backprop
# print("Backprop")
# softgrad = soft.backward(Y_label)
# print("Softmax Grad ",softgrad.shape)
# delta, dw, db = FullyConnected.backward(softgrad)
# print("Fully Connected Grad ",delta.shape)
# #FullyConnected.updateParams(dw,db,0.01)
# flat_grad = flat.backward(delta)
# print("Flatten Grad ",flat_grad.shape)
# relu3_grad = relu3.backward(flat_grad)
# print("Relu3 Grad ",relu3_grad.shape)
# conv3_grad,conv3_dw, conv3_db = conv3.backward(relu3_grad)
# #conv3.update_params(conv3_dw,conv3_db,0.01)
# print("Conv3 Grad ",conv3_grad.shape)
# maxpool_grad = maxpool2.backward(conv3_grad)
# print("Maxpool Grad ",maxpool_grad.shape)
# relu_grad = relu2.backward(maxpool_grad)
# print("Relu Grad ",relu_grad.shape)
# conv2_grad,conv2_dw, conv2_db = conv2.backward(relu_grad)
# #conv2.update_params(conv2_dw,conv2_db,0.01)
# print("Conv2 Grad ",conv2_grad.shape)
# maxpool_grad = maxpool1.backward(conv2_grad)
# print("Maxpool Grad ",maxpool_grad.shape)
# relu_grad = relu1.backward(maxpool_grad)
# print("Relu Grad ",relu_grad.shape)
# conv1_grad,conv1_dw, conv1_db = testconv.backward(relu_grad)
# #testconv.update_params(conv1_dw,conv1_db,0.01)
# print("Conv1 Grad ",conv1_grad.shape)

# #print(conv1_grad[0])


# print("Cross entropy loss = ",crossEntropy(out,Y_label))

def accuracy(out,y_label):
    outShape = out.shape
    correct = 0
    for i in range(outShape[0]):
        max = 0
        maxIndex = 0
        for j in range(outShape[1]):
            if out[i,j] > max:
                max = out[i,j]
                maxIndex = j
        if maxIndex == y_label[i]:
            correct += 1
    return correct/outShape[0]

class Model:
    def __init__(self):
        self.conv1 = conv(6,5,1,2,1)
        self.relu1 = Relu()
        self.maxpool1 = Maxpool(2,2)
        self.conv2 = conv(12,5,1,0,6)
        self.relu2 = Relu()
        self.maxpool2 = Maxpool(2,2)
        self.conv3 = conv(100,5,1,0,12)
        self.relu3 = Relu()
        self.flat = Flatten()
        self.FullyConnected = FullyConnected(10,100)
        self.soft = Softmax()
        
    def forward(self,input):
        input = np.reshape(input,(input.shape[0],28,28,1))
        input = self.conv1.forward(input)
        input = self.relu1.forward(input)
        input = self.maxpool1.forward(input)
        input = self.conv2.forward(input)
        input = self.relu2.forward(input)
        input = self.maxpool2.forward(input)
        input = self.conv3.forward(input)
        input = self.relu3.forward(input)
        input = self.flat.forward(input)
        input = self.FullyConnected.forward(input)
        input = self.soft.forward(input)
        return input

    def backward(self,grad):
        grad = self.soft.backward(grad)
        grad = self.FullyConnected.backward(grad)
        grad = self.flat.backward(grad)
        grad = self.relu3.backward(grad)
        grad = self.conv3.backward(grad)
        grad = self.maxpool2.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.conv2.backward(grad)
        grad = self.maxpool1.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.conv1.backward(grad)
        return grad
    
    def train_model(self, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, batch_size, epochs, learning_rate):
        for epoch in range(epochs):
            for i in range(0,X_train.shape[0],batch_size):
                X_batch = X_train[i:i+batch_size]
                Y_batch = Y_train[i:i+batch_size]
                Y_label = np.zeros((Y_batch.size, 10))
                Y_label[np.arange(Y_batch.size), Y_batch.ravel()] = 1
               # print("Y_label  ",Y_label.shape," at " ,i)
                out = self.forward(X_batch)
                grad = self.backward(Y_label)
            output1 = self.forward(X_valid)
            output2 = self.forward(X_test)
            print("Epoch ", epoch, " Loss ", crossEntropy(output1, Y_valid))
            print("Epoch ",epoch," Accuracy ",accuracy(output2,Y_test))
            
    


# model = Model()
# x = model.forward(mytrain_images[:32])
# print( " dd ",x.shape)
# x = model.backward(x)
# print(x.shape)
mytrain_images = train_images[0:5000]
mytrain_labels = train_labels[0:5000]

mytest_images = test_images[0:5000]
mytest_labels = test_labels[0:5000]

myvalid_images = train_images[5000:6000]
myvalid_labels = train_labels[5000:6000]


model = Model()
model.train_model(mytrain_images,mytrain_labels,myvalid_images,myvalid_labels,mytest_images,mytest_labels,32,10,0.001)
