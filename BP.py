#coding:utf-8

from math import *
import numpy as np
from numpy import *
import numpy as np
from os import listdir

import test_loadimage
learningrate = 0.01

def loadImages(dirName):
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumstr = int(fileStr.split('_')[0])
        if classNumstr == 1:hwLabels.append([0,1,0,0,0,0,0,0,0,0])
        else:hwLabels.append([0,0,0,0,0,0,0,0,0,1])
        trainingMat[i,:] = img2vector('%s/%s'%(dirName , fileNameStr))
    return trainingMat , hwLabels

def img2vector(filename):
    #将32*32的矩阵转换为1*1024的向量
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline() #一次读一行
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def WeightInit(i,j):
    """
    权值初始化
    :param i:前一层的神经元数量
    :param j: 后一层的神经元数量
    :return: 初始化后的权值矩阵
    """
    W = mat(random.uniform(-2.4 / i, 2.4 / i, size=(j, i)))
    return W

def init(M):
    W = [] ; b = []
    for i in range(1,len(M)):
        W.append(WeightInit(M[i-1] , M[i]))
        b.append(mat(random.uniform(-2.4/M[i],2.4/M[i],size=(1,M[i]))).transpose())
    return W,b

#激励函数
def sigmoid(x):
    if type(x) != int :
        y = np.zeros(shape(x))
        for i in range(len(x)):
            y[i] = 1/(1+exp(-x[i]))
    else:
        y = 1/(1+exp(-x))
    return y

#激励函数的导数
def ft(x):
    y = sigmoid(x)*(1-sigmoid(x))
    return y

def forward(M , W , b , data , label):
    """
    前向推导函数
    :param M: 层数向量
    :param W: 权值矩阵
    :param data: 传入的单个数据
    :param label:传入的单个标签
    :return:net, O, y, E矩阵
    """
    Layer = len(M)
    net = [];O = []
    for i in range(Layer):
        net.append(mat(np.zeros(M[i])).T)
        O.append(mat(np.zeros(M[i])).T)
    O[0] = data.transpose()
    y = mat(np.zeros(10)).T
    E = mat(np.zeros(10)).T
    for m in range(1,Layer):
        net[m] = W[m-1] * O[m-1] + b[m-1]; #第m层神经元净输入
        O[m] = sigmoid(net[m]); #第m层神经元净输出
    y = O[Layer-1];
    E = 0.5*multiply(y-label,y-label)
    return net, O, y, E

def backward(M, W, b, net, O, y, E, label,rate):
    """
    后向推导函数
    :param M:层数向量
    :param W:权值矩阵
    :param net:forward return的每一层的输入,list内嵌列矩阵
    :param O:forward return的每一层的输出,list内嵌列矩阵
    :param y:forword return的该数据的预测输出,列矩阵
    :param E:forward return的预测值和真实值的平方的二分之一
    :return:更新后的权值矩阵
    """
    grad = []
    for i in range(len(M)):
        grad.append(mat(np.zeros(M[i])).transpose())
    layer = list(range(1,len(M)))
    layer.reverse()
    for m in layer: #从输出层回退
        if m == len(M) - 1:  # 如果是输出层
            grad[m] = -multiply((label - y), ft(net[m]))
            W[m-1] -= rate * grad[m] *O[m-1].transpose()
            b[m-1] -= rate * grad[m]
        else:
            t = W[m].transpose()*grad[m+1]
            grad[m] = multiply(ft(net[m]),t)
            W[m-1] -= rate * grad[m]*O[m-1].transpose()
            b[m-1] -= rate * grad[m]
    return  W , b

def training(M , W , b, dataMat , labelMat , testMat , labelMat2 , iteration = 5 ):
    """
    训练模型,输出结果
    :param M:神经元数量向量
    :param W: 初始化的权值矩阵
    :param dataSet: 数据集
    :param labelSet: 标签集
    :return: 训练后的结果
    """
    for iter in range(iteration):
        print("epoc %d:"%iter)
        for i in range(len(dataMat)):
            net, O, y, E = forward(M , W , b , dataMat[i] , labelMat[i].transpose())
            rate = learningrate + 1/(iter + 1)
            W , b = backward(M , W , b, net , O , y ,E , labelMat[i].transpose(),rate)
            #print("第%d个样本训练！"%i)
        error = test(dataMat , labelMat , M , W , b)
        error2 = test(testMat , labelMat2 , M , W , b)
    return W , b

def test(testSet ,testLabel , M , W , b):
    count = 0
    for i in range(len(testSet)):
        net, O, y, E = forward(M , W , b , testSet[i] , testLabel[i].transpose())
        t = argmax(y)
        if t == argmax(testLabel[i].transpose()):
            count += 1
            #print("data%d 测试正确\n" % i)
    print("正确率为：%f" % (count/len(testSet)))
    return count/len(testSet)

def store(input , filename):
    import pickle
    fw = open(filename , 'wb')
    pickle.dump(input , fw)
    fw.close()

def grab(filename):
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)

def show(M , W , b , data ):
    Layer = len(M)
    net = []
    O = []
    for i in range(Layer):
        net.append(mat(np.zeros(M[i])).T)
        O.append(mat(np.zeros(M[i])).T)
    O[0] = data.transpose()
    y = mat(np.zeros(10)).T
    E = mat(np.zeros(10)).T
    for m in range(1, Layer):
        net[m] = W[m - 1] * O[m - 1] + b[m - 1]  # 第m层神经元净输入
        O[m] = sigmoid(net[m])  # 第m层神经元净输出
    y = O[Layer - 1]
    return y

if __name__ == "__main__":
    M = [256, 25 , 10]
    W, b = init(M)
    dataArr, labelArr, testArr, labelArr2 = test_loadimage.getdata()
    dataMat = mat(dataArr)
    labelMat = mat(labelArr)
    testMat = mat(testArr)
    labelMat2 = mat(labelArr2)
    W, b = training(M, W, b, dataMat, labelMat, testMat , labelMat2 , 100)
    store(W, 'weights.txt')
    store(b , 'biases.txt')
    #b = grab('biases.txt')
    test(testMat , labelMat2 , M , W , b)
