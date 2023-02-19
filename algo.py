import numpy as np
import matplotlib.pyplot as plt


#Creating an Algo class which is a simple linear combination of weights and features with a relu activation function
class Algo():
    def __init__(self, X, y):
        '''
        args: X=array; independent variables
              y=array; dependent variable
        '''
        self.X=X
        self.y=y
        self.xshape=np.shape(X)
        self.yshape=np.shape(y)
        #number of rows in X must equal those in y
        assert self.xshape[0]==self.yshape[0]

    def init_weight(self):
        '''
        Initialize weights and bias for algorithm
        returns: w=array; weight matrix
                 b=array; bias
        '''
        # set random seed for consistent results
        np.random.seed(19)
        w=np.random.normal(size=(self.xshape[1]))
        b=np.random.normal(size=1)
        return w,b

    def create_equ(self,w,b):
        '''
        Generate predicted y values using y=w.T x 
        args: w=array; weight matrix
              b=array; bias
        returns: y_pred=array; predicted y values 
        '''
        #Create transpose of X 
        trans_X=np.transpose(self.X)
        y_pred=np.matmul(w,trans_X) + b
        return y_pred

    def mean_squared_error(self, y_pred):  
        '''
        Calculate mean squared error loss function
        args: y_pred=array; predicted y values
        returns: mse=float; mean squared error
        '''
        mse=1/self.xshape[0] *(np.sum(np.square(self.y-y_pred)))
        return mse

    
    
    def gradient_descent(self, y_pred):
        '''
        Gradient descent optimization algorithm to generate derivative of MSE and w and MSE and bias respectively
        args: y_pred=array; predicted y values 
        returns: dW_arr=array; dMSE/dW-partial derivative of MSE with weight matrix
                 dB=array; dMSE/dB- partial derivative of MSE with bias 
        '''
        lis=[]
        #Transpose self.X to aid calculation
        x=np.transpose(self.X)
        for i in x:
            sum=0
            for j in range(self.xshape[0]):
                sum+=np.dot(i[j], (self.y[j]-y_pred[j]))
            dW=(-2/self.xshape[0]) * sum
            lis.append(dW)
        dW_arr=np.array(lis)
        dB=-2/self.xshape[0] *(np.sum((self.y-y_pred)))
        return dW_arr, dB 


    def fit(self, iterations, learn_rate):
        '''
        Brings all methods together to run one iterative method
        args: iterations=int; number of times to run algorithm
              learn_rate=float; learn rate for gradient descent step typically small so that global minima is not missed
        returns: w=array; weight matrix of final model after iterations
                 b=array; bias of final model after iterations
                 lis=list; list of mean squared error after each iteration
        '''
        lis=[]
        w,b=self.init_weight()
        for i in range(iterations):
            y_pred=self.create_equ(w,b)
            mse=self.mean_squared_error(y_pred)
            lis.append(mse)
            dW_arr, dB=self.gradient_descent(y_pred)
            w=w-(learn_rate *dW_arr)
            b=b-(learn_rate * dB)
        return w,b,lis 
    
    def plot_mse(self,iterations,learn_rate):
        '''
        Matplotlib plot of mean squared error against iterations
        args: iterations=int; number of times to run algorithm
              learn_rate=float; learn rate for gradient descent step typically small so that global minima is not missed
        returns: plot of mse against iterations

        '''
        #Return mean squared error from fit method
        w,b,lis=self.fit(iterations,learn_rate)
        plt.figure(figsize=(5,10))
        plt.plot(1,len(lis)+1, lis)
        plt.xlabel('number of iterations')
        plt.ylabel('mean squared error')
        plt.show()







    
