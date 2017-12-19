import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys
from sklearn.metrics import accuracy_score

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix

    # IMPLEMENT THIS METHOD
    classes = np.unique(y)
    k = classes.size
    N, d = X.shape
    # print("N=",N,"d=",d,"k=",k,"classes=",classes)
    means = np.zeros((d,k))
    # vars = np.zeros((d,k))
    covmat = np.zeros((d,d))
    # mimic Matlab cell,group[1] will contain all data with label 1,etc
    group = np.empty(k+1, dtype=object)


    for i in range(k):
        group[int(classes[i])] = np.transpose(X[np.where(y == classes[i])[0]])
        means[:, i] = np.mean(group[i+1], axis=1)
        covmat += (group[i+1].shape[1]-1)*np.cov(group[i+1])
    covmat /= N-k
    # print(means)
    # print(covmat)

    # print(group[1])
    # print(group[1].shape)
    # print(group[2].shape)
    # print(group[3].shape)
    # print(group[4].shape)
    # print(group[5].shape)
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes

    # IMPLEMENT THIS METHOD
    classes = np.unique(y)
    k = classes.size
    N, d = X.shape
    # print("N=",N,"d=",d,"k=",k,"classes=",classes)
    means = np.zeros((d,k))
    # vars = np.zeros((d,k))
    covmats = []
    # mimic Matlab cell,group[1] will contain all data with label 1,etc
    group = np.empty(k+1,dtype=object)

    for i in range(k):
        group[int(classes[i])] = np.transpose(X[np.where(y==classes[i])[0]])
        means[:,i] = np.mean(group[i+1],axis=1)
        covmats.append(np.cov(group[i+1]))

    # print(covmats)
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred N x 1

    # IMPLEMENT THIS METHOD
    N = ytest.size
    classes = np.unique(y)
    d,k = means.shape
    ypred = np.empty((N,1))
    num = 0

    for i in range(N):
        scores = np.zeros((1,k))
        for j in range(k):
            scores[:,j] = (Xtest[i]).dot(inv(covmat)).dot(means[:,j])-1/2*(means[:,j].T).dot(inv(covmat)).dot(means[:,j])

        ypred[i] = classes[np.argmax(scores)]
        if ypred[i] == ytest[i]:
            num +=1

    # print(accuracy_score(ytest,predicts))
    acc = num / N
    # print(acc)
    return acc, ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred

    # IMPLEMENT THIS METHOD
    N = ytest.size
    classes = np.unique(y)
    d,k = means.shape
    ypred = np.empty((N,1))
    num = 0
    for i in range(N):
        scores = np.zeros((1,k))
        for j in range(k):
            scores[:,j] = -1/2*np.log2(det(covmats[j]))-1/2*(Xtest[i]-means[:,j]).dot(inv(covmats[j])).dot((Xtest[i]-means[:,j]).T)
        # print("scores=",scores)
        # shift from 0~k-1 to 1~k
        ypred[i] = classes[np.argmax(scores)]
        if ypred[i] == ytest[i]:
            num +=1
    acc = num / N
    return acc, ypred

def learnOLERegression(X,y):
    # Inputs:
    # X = N x d
    # y = N x 1
    # Output:
    # w = d x 1
    # IMPLEMENT THIS METHOD
    w = inv(X.T.dot(X)).dot(X.T).dot(y)
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d
    # y = N x 1
    # lambd = ridge parameter (scalar)
    # Output:
    # w = d x 1

    # IMPLEMENT THIS METHOD
    N,d = X.shape
    w = inv(lambd*np.eye(d,dtype=int)+X.T.dot(X)).dot(X.T).dot(y)
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = N x 1
    # Output:
    # rmse

    # IMPLEMENT THIS METHOD
    rmse = sqrt(np.mean((ytest-Xtest.dot(w))**2))
    return rmse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda

    # IMPLEMENT THIS METHOD
    w=np.reshape(w,(w.size,1))
    error = 1/2*np.sum((y-X.dot(w))**2)+1/2*lambd*w.T.dot(w)
    error_grad = -X.T.dot(y-X.dot(w))+lambd*w
    error_grad = error_grad.ravel()
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:
    # x - a single column vector (N x 1)
    # p - integer (>= 0)
    # Outputs:
    # Xd - (N x (d+1))
    # IMPLEMENT THIS METHOD
    N = x.size
    Xd = np.empty((N,p+1))
    xs = x.tolist()
    r = list(range(p))
    r.append(p)
    for i in range(N):
        Xd[i,:] = [xs[i]**j for j in r]
    return Xd

# Main script

# Problem 1
# load the sample data
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')
# # LDA
means,covmat = ldaLearn(X,y)
ldaacc,pred = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# # QDA
means,covmats = qdaLearn(X,y)
qdaacc,pred = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()
zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.figure()
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.figure()
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)

# Problem 2
#
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)
#
w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))
#
# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.figure()
plt.plot(lambdas,rmses3)

# lambdas = 1
# w_l = learnRidgeRegression(X_i,y,lambdas)
# err = sqrt(np.mean((Xtest_i.dot(w_l) - ytest) ** 2))
# print(err)

# # Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    rmses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.figure()
plt.plot(lambdas,rmses4)
#
#
# # Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
plt.figure()
plt.plot(range(pmax),rmses5)
plt.legend(('No Regularization','Regularization'))
plt.show()

