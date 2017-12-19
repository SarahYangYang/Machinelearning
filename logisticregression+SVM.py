import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.svm import SVC
import pickle


def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation
    # print("n_sample=",n_sample)
    # print("n_train=",n_train)
    # print("n_validation=",n_validation*10)


    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    # print("index=",type(index[0]))
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    # According to the given formula (2)
    # We now have N<==>n_data, y_n <==> labeli, the only thing we need is theta
    # theta = np.zeros((n_train,1))
    # Add bias to train_data
    train = np.hstack((np.ones((n_data,1)),train_data))
    theta = sigmoid(np.dot(train,initialWeights))
    theta = np.reshape(theta,(n_train,1))

    # print("labeli shape=",labeli.shape)
    # print("theta shape=",theta.shape)
    # print("Start computing current error")
    error = -1.0/n_data*np.sum(labeli*np.log(theta)+(1-labeli)*np.log(1-theta))
    delta = (theta-labeli)*train
    # print("Start computing current grad")
    error_grad = 1.0/n_data*np.sum(delta,axis=0)
    print("error=",error)
    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    n_data = data.shape[0]
    train = np.hstack((np.ones((n_data,1)),data))
    numeric_value = np.dot(train,W)
    label=np.argmax(numeric_value,axis=1).reshape(n_data,1)
    print(label)
    return label

# def softmax(x):
#     """Compute softmax values for each sets of scores in x."""
#     # print("input of softmax=",x)
#     result = np.exp(1.0*x) / np.sum(np.exp(1.0*x))
#     # print("output of softmax=",result)
#     return result
#
def softmax(x):
    e = np.exp(x - np.max(x),dtype=np.float64)  # prevent overflow

    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:
        result = e / np.array([np.sum(e)])
        # if any(i == -np.inf for i in result):
        #     print("i=",i)
        return result  # ndim = 2



def mlrObjFunction(initialWeights, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data,Y = args

    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
# According to the given formula (2)
    # We now have N<==>n_data, y_n <==> labeli, the only thing we need is theta
    # theta = np.zeros((n_train,1))
    # Add bias to train_data
    train = np.float64(np.hstack((np.ones((n_data,1)),train_data)))
    # print("data shape=",train.shape)
    # print("weights shape=",initialWeights.shape)
    # print("12th of initialWeights=",initialWeights[12])
    initialWeights = np.float64(np.reshape(initialWeights,(n_feature+1,n_class)))
    product = np.float64(np.dot(train,initialWeights))
    theta = np.zeros((n_train,n_class),dtype=np.float64)
    for i in range(n_train):
        theta[i,:] = softmax((product[i,:]))
    # print("theta first row=",theta[0,:])
    # theta = np.reshape(theta,(n_train,10))
    # theta = np.float64(theta)
    # print("labeli shape=",labeli.shape)
    # print("theta shape=",theta.shape)
    # print("Start computing current error")
    # if np.any(theta==0):
    #     print("theta=",theta)
    result = np.log(theta)
    result[result==-np.inf]=-10**16
    error = -np.sum(Y*result)/n_data
    delta = theta-Y
    # print("delta[:,i] shape=",delta[:,0].shape)
    # print("train shape=",train.shape)
    for i in range(n_class):
        delta_ith = np.reshape(delta[:,i],(n_train,1))
        temp = delta_ith*train
        # print("temp shape",temp.shape)
        error_grad[:,i] = np.sum(temp,axis=0)
    # error_grad = np.reshape(error_grad,(n_class*(n_feature+1),1))
    error_grad = (error_grad/n_data).flatten()
    print(error)
    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    n_data = data.shape[0]
    train = np.hstack((np.ones((n_data,1)),data))
    numeric_value = np.dot(train,W)
    label=np.argmax(numeric_value,axis=1).reshape(n_data,1)
    return label


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
print("Preprocess Done")
# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 50}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    #print("Start computing for class ",i)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')



"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################
def SVM_linear(*args):
    X_train,y_train,X_val,y_val,X_test,y_test = args
    y_train = y_train.ravel()
    y_val = y_val.ravel()
    y_test = y_test.ravel()
    linear_svc = SVC(C=1.0, kernel='linear', degree=3, gamma='auto', coef0=0.0, shrinking=True,
          probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False,
          max_iter=-1, random_state=None)
    print("SVC Linear Train start...")
    linear_svc.fit(X_train,y_train)
    print("Done...")
    # Training Accuracy
    print("train accuracy start...")
    train_predicts = linear_svc.predict(X_train)
    print("Train done...")
    train_accuracy = np.sum(train_predicts==y_train)/y_train.size
    # Validation Accuracy
    print("validation accuracy start...")
    val_predicts = linear_svc.predict(X_val)
    print("validation done...")
    val_accuracy = np.sum(val_predicts==y_val)/y_val.size
    # Test Accuracy
    print("test accuracy start...")
    test_predicts = linear_svc.predict(X_test)
    print("test done...")
    test_accuracy = np.sum(test_predicts==y_test)/y_test.size
    return train_accuracy,val_accuracy,test_accuracy
def SVM_radial_gamma1(*args):
    X_train,y_train,X_val,y_val,X_test,y_test = args
    y_train = y_train.ravel()
    y_val = y_val.ravel()
    y_test = y_test.ravel()
    radial_svc = SVC(kernel='rbf',gamma=1.0)
    print("SVC Radial Train start...")
    radial_svc.fit(X_train,y_train)
    print("Done...")
    # Training Accuracy
    print("train accuracy start...")
    train_predicts = radial_svc.predict(X_train)
    print("Train done...")
    train_accuracy = np.sum(train_predicts==y_train)/y_train.size
    # Validation Accuracy
    print("validation accuracy start...")
    val_predicts = radial_svc.predict(X_val)
    print("validation done...")
    val_accuracy = np.sum(val_predicts==y_val)/y_val.size
    # Test Accuracy
    print("test accuracy start...")
    test_predicts = radial_svc.predict(X_test)
    print("test done...")
    test_accuracy = np.sum(test_predicts==y_test)/y_test.size
    return train_accuracy,val_accuracy,test_accuracy

def SVM_radial_gammaDefault(*args):
    X_train,y_train,X_val,y_val,X_test,y_test = args
    y_train = y_train.ravel()
    y_val = y_val.ravel()
    y_test = y_test.ravel()
    radial_svc_gammaDefault = SVC(kernel='rbf')
    print("SVC Radial with default gamma Train start...")
    radial_svc_gammaDefault.fit(X_train,y_train)
    print("Done...")
    # Training Accuracy
    print("train accuracy start...")
    train_predicts = radial_svc_gammaDefault.predict(X_train)
    print("Train done...")
    train_accuracy = np.sum(train_predicts==y_train)/y_train.size
    # Validation Accuracy
    print("validation accuracy start...")
    val_predicts = radial_svc_gammaDefault.predict(X_val)
    print("validation done...")
    val_accuracy = np.sum(val_predicts==y_val)/y_val.size
    # Test Accuracy
    print("test accuracy start...")
    test_predicts = radial_svc_gammaDefault.predict(X_test)
    print("test done...")
    test_accuracy = np.sum(test_predicts==y_test)/y_test.size
    return train_accuracy,val_accuracy,test_accuracy
def SVM_radial_varingC(*args):
    train_accuracys = []
    val_accuracys = []
    test_accuracys = []

    X_train,y_train,X_val,y_val,X_test,y_test = args
    y_train = y_train.ravel()
    y_val = y_val.ravel()
    y_test = y_test.ravel()

    for i in range(0,110,10):
        if i==0:
            i=1.0
        else:
            i=i*1.0
        radial_svc_varingC = SVC(C=i,kernel='rbf')
        print("SVC Radial with default gamma Train start...")
        radial_svc_varingC.fit(X_train,y_train)
        print("Done...")
        # Training Accuracy
        print("train accuracy start...")
        train_predicts = radial_svc_varingC.predict(X_train)
        print("Train done...")
        train_accuracy = np.sum(train_predicts==y_train)/y_train.size
        train_accuracys.append(train_accuracy)
        # Validation Accuracy
        print("validation accuracy start...")
        val_predicts = radial_svc_varingC.predict(X_val)
        print("validation done...")
        val_accuracy = np.sum(val_predicts==y_val)/y_val.size
        val_accuracys.append(val_accuracy)
        # Test Accuracy
        print("test accuracy start...")
        test_predicts = radial_svc_varingC.predict(X_test)
        print("test done...")
        test_accuracy = np.sum(test_predicts==y_test)/y_test.size
        test_accuracys.append(test_accuracy)
    return train_accuracys,val_accuracys,test_accuracys

a1,b1,c1 = SVM_linear(train_data, train_label, validation_data, validation_label, test_data, test_label)
print("For SVM linear, train accuracy:{} validation accuracy:{} test accuracy:{}".format(a1,b1,c1))
a2,b2,c2 = SVM_radial_gamma1(train_data, train_label, validation_data, validation_label, test_data, test_label)
print("For SVM radial with gamma equals 1, train accuracy:{} validation accuracy:{} test accuracy:{}".format(a2,b2,c2))
a3,b3,c3 = SVM_radial_gammaDefault(train_data, train_label, validation_data, validation_label, test_data, test_label)
print("For SVM radial with gamma equals default, train accuracy:{} validation accuracy:{} test accuracy:{}".format(a3,b3,c3))
a4,b4,c4 = SVM_radial_varingC(train_data, train_label, validation_data, validation_label, test_data, test_label)
for i in range(len(a4)):
    print("For SVM linear, train accuracy:{} validation accuracy:{} test accuracy:{}".format(a4[i],b4[i],c4[i]))

#
"""
Script for Extra Credit Part
"""
print('\n\n--------------Multi-class Logistic Regression-------------------\n\n')
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class),dtype=np.float64)
# initialWeights_b[1,2] = 99
opts_b = {'maxiter': 50}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
parameters = {}
parameters_bonus = {}
parameters['W']=W
parameters_bonus={'W_b':W_b}

pickle.dump(parameters, open("params.p", "wb"))
pickle.dump(parameters_bonus, open("params_bonus.p", "wb"))
