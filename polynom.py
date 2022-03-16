#from numpy.polynomial import Polynomial as np
import numpy as np
from pandas import read_csv as pd
import matplotlib.pyplot as plt
import sys
import time

csv = pd('./result.csv')
data = csv[['Row', 'TTD', 'UnixTimestamp']]
x = data['Row']
y = data['TTD']
t = data['UnixTimestamp']
l=int(len(y))-1

csv_train = pd('./result.csv', nrows=(int(l/2)))

data_train = csv_train[['Row', 'TTD']]
xt = data['Row']
yt = data['TTD']

degree=3
diff=1
step=np.average(np.diff(t))

predict = np.polyfit(x, y, degree)
predict_time = np.polyfit(x, t, degree)

plt.scatter(x, y) 
p = np.poly1d(predict)
plt.plot(x,p(x),"r--")

plt.savefig('chart.png')

i=0
while i <= degree:

    if i == degree:
        print("(%.10f)"%(predict[i]))
        break
    print("(%.10f*(ROW(A2)-1)^%d)+"%(predict[i],degree-i,),end="")
    i+=1

def estimate_ttd(target):
    copy=np.copy(predict)
    copy[-1] -= target
    point=int((np.roots(copy)[degree-1]))
    if point <= l:
        print(time.ctime(t[point]))
    else:
        er=(abs(((point - l)*step+t[l])-np.polyval(predict_time,point)))
        mid=(((point - l)*step+t[l])+np.polyval(predict_time,point))/2
        print("TTD of", target, " is expected between",time.ctime(mid-er),"and",time.ctime(mid+er))

if (len(sys.argv)) > 1:
    target = int(sys.argv[1])
    estimate_ttd(target)

#plt.show()
