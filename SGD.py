#from random import seed
from random import randrange
from csv import reader
from math import sqrt

#加载一个CSV文件
def load_csv(filename):
	dataset = list()
	with open(filename,'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset
	
#转换string列为float类型
def str_column_to_float(dataset,column): #将数据集的第column列转换成float形式
	for row in dataset:
		row[column] = float(row[column].strip()) #strip（）返回移除字符串头尾指定的字符生成的新字符串


#为每列找到最小值和最大值
def dataset_minmax(dataset):
	minmax = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min,value_max])
	return minmax
	
#调整列数据范围为0-1之间
def normlize_dataset(dataset,minmax):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i]-minmax[i][0])/(minmax[i][1]-minmax[i][0])
			
#将数据集dataset分成n_flods份，每份包含len(dataset) / n_folds个值，每个值由dataset数据集的内容随机产生，每个值被使用一次
def cross_validation_split(dataset,n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)  #复制一份dataset,防止dataset的内容改变
	fold_size = len(dataset)/n_folds
	for i in range(n_folds):
		fold = list() #每次循环fold清零，防止重复导入dataset_split
		while len(fold) < fold_size: #这里不能用if，if只是在第一次判断时起作用，while执行循环，直到条件不成立
			index = randrange(len(dataset_copy))
			#将对应索引index的内容从dataset_copy中导出，并将该内容从dataset_copy中删除。
			#pop() 函数用于移除列表中的一个元素（默认最后一个元素），并且返回该元素的值。
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split   #由dataset分割出的n_folds个数据构成的列表，为了用于交叉验证

#计算标准差
def rmse_metric(actual,predicted): 
	sum_error = 0.0
	for i in range(len(actual)):
		prediction_error = predicted[i]-actual[i]
		sum_error += (prediction_error**2)
	mean_error = sum_error/float(len(actual))
	return sqrt(mean_error)
	
#使用交叉验证来评估一个算法
def evaluate_algorithm(dataset,algorithm,n_folds,*args): #评估算法性能，返回模型得分
	folds = cross_validation_split(dataset,n_folds)
	scores = list()
	for fold in folds: #每次循环从folds从取出一个fold作为测试集，其余作为训练集，遍历整个folds，实现交叉验证
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set,[]) #将多个fold列表组合成一个train_set列表
		test_set = list()
		for row in fold: #fold表示从原始数据集dataset提取出来的测试集
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set,test_set,*args)
		actual = [row[-1] for row in fold]
		rmse = rmse_metric(actual,predicted)
		scores.append(rmse)
	return scores

#用系数进行预测
def predict(row,coefficients):
	yhat = coefficients[0]
	for i in range(len(row)-1):
		yhat += coefficients[i+1]*row[i]
	return yhat

#使用随机梯度下降法评价线性回归的有效性
def coefficients_sgd(train,l_rate,n_epoch):
		coef = [0.0 for i in range(len(train[0]))]
		for epoch in range(n_epoch):
			for row in train:
				yhat = predict(row,coef)
				error = yhat - row[-1]
				coef[0] = coef[0] - l_rate*error
				for i in range(len(row)-1):
					coef[i+1] = coef[i+1] - l_rate*error*row[i]
				print(l_rate,n_epoch,error)
		return coef

#利用随机梯度下降法估算线性回归系数
def linear_regression_sgd(train,test,l_rate,n_epoch):
	predictions = list()
	coef = coefficients_sgd(train,l_rate,n_epoch)
	for row in test:
		yhat = predict(row,coef)
		predictions.append(yhat)
	return(predictions)
	
#葡萄酒质量数据集的线性回归
#seed(1)   #每一次执行本文件时都能产生同一个随机数
#载入准备的数据
filename = 'D:\python\winequality-white.csv'
dataset =load_csv(filename)
for i in range(len(dataset[0])):
	str_column_to_float(dataset,i)
#统一化
minmax = dataset_minmax(dataset)
normlize_dataset(dataset,minmax)
#评估算法
n_folds = 5 #分成5份数据，进行交叉验证
l_rate = 0.01
n_epoch = 50
scores = evaluate_algorithm(dataset,linear_regression_sgd,n_folds,l_rate,n_epoch)
print('Scores: %s' % scores)
print('Mean RMSE：%.3f' % (sum(scores)/float(len(scores))))
	

















	