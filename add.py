from numpy import * 
import matplotlib.pyplot as plt

#X=[4,8,9,8,7,12,6,10,6,9]
#Y=[9,20,22,15,17,23,18,25,10,20]
X=[15.50,23.75,8.00,17.00,5.50,19.00,24.00,2.50,7.50,11.00,13.00,3.75,25.00,9.75,22.00,18.00,6.00,12.50,2.00,21.50]
Y=[2158.70,1678.15,2316.00,2061.30,2207.50,1708.30,1784.70,2575.00,2357.90,2256.70,2165.20,2399.55,1779.80,2336.75,
1765.30,2053.50,2414.40,2200.50,2654.20,1753.70]

Xavg = 0
Yavg = 0

for x in X:
	Xavg = Xavg + x
Xavg = Xavg / 20

for y in Y:
	Yavg = Yavg + y
Yavg = Yavg / 20

Lxy = 0
Lxx = 0

for i in range(0,20):
	Lxy = Lxy + (X[i]-Xavg)*(Y[i]-Yavg)
	Lxx = Lxx + (X[i]-Xavg)*(X[i]-Xavg)

b = Lxy / Lxx
a = Yavg -b*Xavg
print(a,b)
plt.xlim(0,26)
plt.ylim(1600,2700)
plt.scatter(X,Y,c = 'r',marker = 'o')

x1 = [X[0],a+b*Y[0]]
y1 = [X[19],a+b*Y[19]]
plt.plot(x1,y1)
plt.show()
	

