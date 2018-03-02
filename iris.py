import csv
import random
import math
import matplotlib.pyplot as plt

N = 0 # Total data
alpha = 0.1 # Change to try different learning rate
irisClass = []
x, theta, bias = ([], [random.random() for _ in range(4)], random.random())
# x[0] = sepal_length
# x[1] = sepal_width
# x[2] = petal_length
# x[3] = petal_width

def __readData__(filePath):
    global N
    with open(filePath) as csvData:
        reader = csv.reader(csvData)
        for rowData in reader:
            x.append([float(rowData[0]), float(rowData[1]), float(rowData[2]), float(rowData[3])])
            irisClass.append(0 if rowData[4]=='Iris-setosa' else 1) # 0 for iris-setosa, 1 for iris-versicolor
    N = len(irisClass)

def __sigmoidFunction__(prediction):
    return 1.0/(1+math.exp(-prediction))

def __targetFunction__(x_i, theta, bias):
    ans = 0.0
    for i in range(len(x_i)):
        ans += x_i[i] * theta[i]
    ans += bias
    return ans

def __updateTheta__(x_i, prediction, fact):
    for i in range(len(theta)):
        theta[i] += alpha * 2 * (fact-prediction) * (1-prediction) * prediction * x_i[i]

def __classIndentifier__(prediction):
    return 0 if prediction < 0.5 else 1

def __batch__():
    error = 0.0
    for i in range(N):
        total = __targetFunction__(x[i], theta, bias)
        prediction = __sigmoidFunction__(total)
        error += (prediction-irisClass[i]) ** 2
        __updateTheta__(x[i], prediction, irisClass[i])
    return error/N

def __plotAccuracy__(axis_x, axis_y):
    plt.plot(axis_x, axis_y)
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.show()

def __SGD__():
    accuracy = [__batch__() for _ in range(60)]
    # print(accuracy)
    __plotAccuracy__([_ for _ in range(len(accuracy))], accuracy)

def __predict__():
    while True:
        sepal_length, sepal_width, petal_length, petal_width = map(float, input().split(','))
        total = __targetFunction__([sepal_length, sepal_width, petal_length, petal_width], theta, bias)
        prediction = __sigmoidFunction__(total)
        print('iris-setosa' if __classIndentifier__(prediction) == 0 else 'iris-versicolor')

def __main__():
    __readData__('iris.data')
    __SGD__()
    __predict__()

__main__()