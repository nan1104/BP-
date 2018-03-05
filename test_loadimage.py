# coding:utf-8
import numpy as np
import sys, os
from PIL import Image, ImageDraw

#将512*512矩阵pre转化成n*n的post矩阵
def MatrixProcess(n,pre,post):
    m=512/n
    m=int(m)
#将矩阵中的值变成0或1(0则为1，255则为0)
    for i in range(512):
        for j in range(512):
            if pre[i, j] == 0:
                pre[i, j] = 1
            else:
                pre[i, j] = 0
    for i in range(0,n):
        for j in range(0,n):
            # print (pre[j * m:(j + 1) * m, i * m:(i + 1) * m])
            # print (sum(pre[j * m:(j + 1) * m, i * m:(i + 1) * m]))
            if np.sum(pre[i*m:(i+1)*m,j*m:(j+1)*m])>=10:
                post[i,j]=1
            else:
                post[i,j]=0
    return post


# 二值判断,如果确认是噪声,用改点的上面一个点的灰度进行替换
# 该函数也可以改成RGB判断的,具体看需求如何

def getPixel(image, x, y, G, N):#降噪
    L = image.getpixel((x, y))
    if L > G:
        L = True
    else:
        L = False

    nearDots = 0
    if L == (image.getpixel((x - 1, y - 1)) > G):
        nearDots += 1
    if L == (image.getpixel((x - 1, y)) > G):
        nearDots += 1
    if L == (image.getpixel((x - 1, y + 1)) > G):
        nearDots += 1
    if L == (image.getpixel((x, y - 1)) > G):
        nearDots += 1
    if L == (image.getpixel((x, y + 1)) > G):
        nearDots += 1
    if L == (image.getpixel((x + 1, y - 1)) > G):
        nearDots += 1
    if L == (image.getpixel((x + 1, y)) > G):
        nearDots += 1
    if L == (image.getpixel((x + 1, y + 1)) > G):
        nearDots += 1

    if nearDots < N:
        return image.getpixel((x, y - 1))
    else:
        return None

    # 降噪


# 根据一个点A的RGB值，与周围的8个点的RBG值比较，设定一个值N（0 <N <8），当A的RGB值与周围8个点的RGB相等数小于N时，此点为噪点
# G: Integer 图像二值化阀值
# N: Integer 降噪率 0 <N <8
# Z: Integer 降噪次数
# 输出
#  0：降噪成功
#  1：降噪失败
def clearNoise(image, G, N, Z):#降噪
    draw = ImageDraw.Draw(image)

    for i in range(0, Z):
        for x in range(1, image.size[0] - 1):
            for y in range(1, image.size[1] - 1):
                color = getPixel(image, x, y, G, N)
                if color != None:
                    draw.point((x, y), color)


# import scipy
#将图片转换成512*512的矩阵
def loadImage(filename,data):
    # 读取图片
    im = Image.open(filename)

    # 显示图片

    im = im.resize((512,512),Image.ANTIALIAS)
    #im.show()
    im = im.convert("L")
    im = im.convert("1")


    #clearNoise(im, 50, 2, 4)#除噪
    data = im.getdata()
    data = np.matrix(data)
    # 变换成512*512
    data = np.reshape(data,(512,512))


    #图片切割
    flag=0
    minx=512
    miny=512
    maxx=0
    maxy=0
    tempminx = 0
    tempminy = 0
    tempmaxx = 512
    tempmaxy = 512
    for i in range(512):
        for j in range(512):
            if data[i,j]!=255:
                flag+=1
                if flag==1:
                    tempminx=i
                    tempminy=j
                tempmaxx=i
                tempmaxy=j
        if flag!=0:
            flag=0
            if tempminx<minx:
                minx=tempminx
            if tempminy<miny:
                miny=tempminy
            if tempmaxx>maxx:
                maxx=tempmaxx
            if tempmaxy>maxy:
                maxy=tempmaxy
    if maxy-miny<40:
        maxy=maxy+25
        miny=miny-25
    if maxx-minx<40:
        maxx=maxx+25
        minx=minx-25
    scalim=Image.fromarray(data[minx:maxx,miny:maxy])
    scalim = scalim.resize((512, 512), Image.ANTIALIAS)
    scalim.show()
    scalim = scalim.convert("L")
    scalim = scalim.convert("1")

    # clearNoise(im, 50, 2, 1)#除噪
    data_s = scalim.getdata()
    data_s = np.matrix(data_s)
    # 变换成512*512
    data_s = np.reshape(data_s, (512, 512))




    #print(data[0])
    #new_im = Image.fromarray(data)
    #new_im.show()
    return data_s


#将txt中数据输出为1xn*n的矩阵，值矩阵，值，txt命名格式'classsNumstr_order'，如'5_2'为第二张5的手写图片转化成的txt
def loadtxt(n,dirName):
    trainingFileList = os.listdir(dirName)
    m = len(trainingFileList)
    trainingMat = np.zeros((m,n*n))
    hwLabels = np.mat(np.zeros((m, 10)))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumstr = int(fileStr.split('_')[0])
        hwLabels[i,int(classNumstr)]=1
        trainingMat[i,:] =img2vector(n,'%s/%s'%(dirName , fileNameStr))
    return trainingMat , hwLabels

#读取txt中的数据
def img2vector(n,filename):
    returnVect = np.zeros((1,n*n))
    fr = open(filename)
    lineStr=fr.readline()
    for i in range(n*n):
        returnVect[0,i] = int(lineStr[i])
    return returnVect


def getimages(n=16):
    trainingimageList = os.listdir("image")
    m = len(trainingimageList)
    for i in range(m):
        data = np.mat(np.zeros((1,512*512)))
        data = loadImage('image/%s'%trainingimageList[i],data)#将图片转换成512*512的矩阵
        data_norm = np.mat(np.zeros((n,n)))#data_norm是将data处理后矩阵
        data_norm = MatrixProcess(n, data, data_norm)
        print (data_norm)
        fileNameStr = trainingimageList[i]
        fileStr = fileNameStr.split('.')[0]
        np.savetxt('data/%s.txt'%fileStr,data_norm,'%d',newline='',delimiter='')
    print('转换完毕')

def getdata(n=16):
    DataSet, Labels = loadtxt(n, "data")
    testSet , testLabels = loadtxt(n,'testdata')
    return DataSet,Labels , testSet , testLabels

def show_loadtxt(n,dirName):
    trainingFileList = os.listdir(dirName)
    m = len(trainingFileList)
    trainingMat = np.zeros((m, n * n))
    hwLabels = np.mat(np.zeros((m, 10)))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        trainingMat[i, :] = img2vector(n, '%s/%s' % (dirName, fileNameStr))
    return trainingMat

def show(n=16):
    data = np.mat(np.zeros((1, 512 * 512)))
    data = loadImage('showimage/show.jpg', data)  # 将图片转换成512*512的矩阵
    data_norm = np.mat(np.zeros((n, n)))  # data_norm是将data处理后矩阵
    data_norm = MatrixProcess(n, data, data_norm)
    print (data_norm)
    np.savetxt('showdata/show.txt' , data_norm, '%d', newline='', delimiter='')

