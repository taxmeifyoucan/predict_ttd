import numpy as np
from pandas import read_csv as pd
import matplotlib.pyplot as plt
import sys


csv = pd('./result.csv')
data = csv[['Row', 'TTD', 'UnixTimestamp']]
x = data['Row']
y = data['TTD']
t = data['UnixTimestamp']
l=int(len(y))-2

csv_train = pd('./result.csv', nrows=(int(l/2)))

data_train = csv_train[['Row', 'TTD']]
xt = data['Row']
yt = data['TTD']

target = int(sys.argv[1])

degree=3

z = np.polyfit(x, y, degree)

z[-1] -= target
print (np.roots(z)[degree-1])

i=0
while i <= degree:

    if i == degree:
        print("(%.10f)"%(z[i]))
        break
    print("(%.10f*(ROW(A2)-1)^%d)+"%(z[i],degree-i,),end="")
    i+=1


plt.scatter(x, y) 
p = np.poly1d(z)
plt.plot(x,p(x),"r--")

plt.savefig('chart.png')
#plt.show()

