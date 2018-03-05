import csv
import random
import math
import matplotlib.pyplot as plt

N = 0 # Total data
epoch = 60
nTraining, nValidation = (0, 0)
alpha = 0.1 # Change to try different learning rate
irisClass = []
x, theta, bias = ([], [random.random() for _ in range(4)], random.random())
# x[0] = sepal_length
# x[1] = sepal_width
# x[2] = petal_length
# x[3] = petal_width

def __readData__(filePath):
    global N, nTraining, nValidation
    with open(filePath) as csvData:
        reader = csv.reader(csvData)
        for rowData in reader:
            x.append([float(rowData[i]) for i in range(len(rowData)-1)])
            irisClass.append(0 if rowData[4]=='Iris-setosa' else 1) # 0 for iris-setosa, 1 for iris-versicolor
    N = len(irisClass)
    nTraining = int(.8 * N)
    nValidation = N - nTraining

def __sigmoidFunction__(prediction):
    return 1.0/(1 + math.exp(-prediction))    

def __targetFunction__(x_i, theta, bias):
    ans = 0.0
    for i in range(len(x_i)):
        ans += x_i[i] * theta[i]
    ans += bias
    return ans

def __lossFunction__(prediction, fact):
    return (fact-prediction) ** 2

def __deltaFunction__(x_i, prediction, fact):
    return 2 * (fact-prediction) * (1-prediction) * prediction * x_i 

def __updateTheta__(x_i, prediction, fact):
    for i in range(len(theta)):
        theta[i] += alpha * __deltaFunction__(x_i[i], prediction, fact)

def __updateBias__(prediction, fact):
    global bias
    bias += alpha * __deltaFunction__(1, prediction, fact)

def __updateFunction__(x_i, prediction, fact):
    __updateTheta__(x_i, prediction, fact)
    __updateBias__(prediction, fact)

def __classIndentifier__(prediction):
    return 0 if prediction < 0.5 else 1

def __training__():
    error = 0.0
    for i in range(nTraining):
        total = __targetFunction__(x[i], theta, bias)
        prediction = __sigmoidFunction__(total)
        error += __lossFunction__(prediction, irisClass[i])
        __updateFunction__(x[i], prediction, irisClass[i])
    return error / nTraining

def __validation__():
    error = 0.0
    for i in range(nTraining, nTraining + nValidation):
        total = __targetFunction__(x[i], theta, bias)
        prediction = __sigmoidFunction__(total)
        error += __lossFunction__(prediction, irisClass[i])
    return error / nValidation

def __plotError__(*printedData):
    for data in printedData:
        plt.plot(data[0], label=data[1])
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.show()

def __SGD__():
    training, validation = ([], [])
    print('Training\tValidation')
    for i in range(epoch):
        training.append(__training__())
        validation.append(__validation__())
        print('%f\t%f' % (training[i], validation[i]))
    __plotError__([training, 'Training'], [validation, 'Validation'])

def __predict__():
    while True:
        sepal_length, sepal_width, petal_length, petal_width = map(float, input().split(','))
        total = __targetFunction__([sepal_length, sepal_width, petal_length, petal_width], theta, bias)
        prediction = __sigmoidFunction__(total)
        print('iris-setosa' if __classIndentifier__(prediction) == 0 else 'iris-versicolor')

def __main__():
    print('theta =', theta)
    print('bias =', bias)
    __readData__('iris.data')
    __SGD__()
    # __predict__()

__main__()
