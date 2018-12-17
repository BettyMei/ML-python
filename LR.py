import matplotlib.pyplot as plt
import numpy as np
import random

#加载数据
def loadDataSet():
	dataMat = [] #创建数据列表
	labelMat = [] #创建标签列表
	fr = open('testSet.txt') #打开文件
	for line in fr.readlines(): #逐行读取
		lineArr = line.strip().split() #去回车，放入列表
		dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])]) #添加数据
		labelMat.append(int(lineArr[2])) #添加标签
	fr.close() #关闭文件
	return dataMat,labelMat

#sigmoid函数
def sigmoid(inX):
	return 1.0/(1+np.exp(-inX))
	
#梯度上升法
def gradAscent(dataMatIn,classLabels):
	dataMatrix = np.mat(dataMatIn) #转换成numpy的mat
	labelMat = np.mat(classLabels).transpose() #转换成numpy的mat,并进行转置
	m,n = np.shape(dataMatrix) #返回dataMatrix的大小，m为行数，n为列数
	alpha = 0.01 #移动步长，用来控制更新的幅度
	maxCycles = 500 #最大迭代次数
	weights = np.ones((n,1)) #每个回归系数初始化为1
	weights_array = np.array([])
	for k in range(maxCycles): #梯度上升矢量化公式
		h = sigmoid(dataMatrix*weights)
		error = labelMat - h
		weights = weights + alpha*dataMatrix.transpose()*error
		weights_array = np.append(weights_array,weights)
	weights_array = weights_array.reshape(maxCycles,n)
	return weights.getA(),weights_array #将矩阵转换为数组,返回权重数组 weights.getA()求得的权重数组(最优参数), weights_array 每次更新的回归系数

#随机梯度上升法
def stocGradAscent(dataMatrix,classLabels,numIter=150):
	m,n = np.shape(dataMatrix) #返回dataMatrix的大小，m为行数，n为列数
	weights = np.ones(n)  #参数初始化
	weights_array = np.array([])  #存储每次更新的回归系数
	for j in range(numIter):
		dataIndex = list(range(m))
		for i in range(m):
			alpha = 4/(1.0+j+i)+0.01   #降低alpha的大小，每次减小1/(j+i)
			randIndex = int(random.uniform(0,len(dataIndex))) #随机选取样本
			h = sigmoid(sum(dataMatrix[randIndex]*weights)) #选择随机选取的一个样本，计算h
			error = classLabels[randIndex]-h #计算误差
			weights = weights+alpha*error*dataMatrix[randIndex]  #更新回归系数
			weights_array = np.append(weights_array,weights,axis=0)  #添加回归系数到数组中
			del(dataIndex[randIndex]) #删除已经使用的样本
	weights_array = weights_array.reshape(numIter*m,n) #改变纬度
	return weights,weights_array   #weights - 求得的回归系数数组(最优参数) weights_array - 每次更新的回归系数
 	
	
#绘制回归系数与迭代次数的关系
def plotWeights(weights_array1,weights_array2):
	fig,axs = plt.subplots(nrows=3,ncols=2,sharex=False,sharey=False,figsize=(13,8))   #当nrow=3,nclos=2时,代表fig画布被分为六个区域,axs[0][0]表示第一行第一列
	x1 = np.arange(0,len(weights_array1),1)
	#绘制w0与迭代次数的关系
	axs[0][0].plot(x1,weights_array1[:,0])
	axs0_title_text = axs[0][0].set_title(u'Stochastic Gradient Ascent Algorithm')
	axs0_ylabel_text = axs[0][0].set_ylabel(u'W0')
	plt.setp(axs0_title_text,size=18,color='black')
	plt.setp(axs0_ylabel_text,size=18,color='black')
	#绘制w1与迭代次数的关系
	axs[1][0].plot(x1,weights_array1[:,1])
	axs1_ylabel_text = axs[1][0].set_ylabel(u'W1')
	plt.setp(axs1_ylabel_text,size=18,color='black')
	#绘制w2与迭代次数的关系
	axs[2][0].plot(x1,weights_array1[:,2])
	axs2_xlabel_text = axs[2][0].set_xlabel(u'Iteration Number')
	axs2_ylabel_text = axs[2][0].set_ylabel(u'W2')
	plt.setp(axs2_xlabel_text,size=18,color='black')
	plt.setp(axs2_ylabel_text,size=18,color='black')
	
	x2 = np.arange(0,len(weights_array2),1)
	#绘制w0与迭代次数的关系
	axs[0][1].plot(x2,weights_array2[:,0])
	axs0_title_text = axs[0][1].set_title(u'Gradient Ascent Algorithm')
	axs0_ylabel_text = axs[0][1].set_ylabel(u'W0')
	plt.setp(axs0_title_text,size=18,color='black')
	plt.setp(axs0_ylabel_text,size=18,color='black')
	#绘制w1与迭代次数的关系
	axs[1][1].plot(x2,weights_array2[:,1])
	axs1_ylabel_text = axs[1][1].set_ylabel(u'W1')
	plt.setp(axs1_ylabel_text,size=18,color='black')
	#绘制w2与迭代次数的关系
	axs[2][1].plot(x2,weights_array2[:,2])
	axs2_xlabel_text = axs[2][1].set_xlabel(u'Iteration Number')
	axs2_ylabel_text = axs[2][1].set_ylabel(u'W2')
	plt.setp(axs2_xlabel_text,size=18,color='black')
	plt.setp(axs2_ylabel_text,size=18,color='black')
	
	plt.show()
	
if __name__=='__main__':
	dataMat,labelMat = loadDataSet()
	weights1,weights_array1 = stocGradAscent(np.array(dataMat),labelMat)
	weights2,weights_array2 = gradAscent(dataMat,labelMat)
	plotWeights(weights_array1,weights_array2)
	

	
	
	
	