#Mess modified old codebase for the merge prediction on Ropsten and printing directly to html. Run with these args for the Ropsten merge setup: --ttd 43531756765713534 --time 1654689600

from re import L
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md
from web3 import Web3
import argparse 
import datetime
import time
import os
import warnings
import datetime as dt


warnings.filterwarnings('ignore')


#Choose web3 provider first, IPC is recommended 
web3 = Web3(Web3.IPCProvider("/home/mario/.ethereum/sepolia/geth.ipc"))

#If result.csv is not present, data will be crawled based on these parameters:
start=12130000 #First block to start with 
granuality=600 #Step in seconds

T = lambda blockn: web3.eth.getBlock(blockn).timestamp
TTD = lambda blockn: web3.eth.getBlock(blockn).totalDifficulty
latest_block = web3.eth.get_block('latest')['number']
start=int(latest_block-(30000))
step=(T(latest_block)-T(start))/(latest_block-start)
degree=3
tolerance=1
counter=0
# Binary search which finds block closest to given timestamp
def block_by_time(timestamp, prev, next):
    global counter
    global tolerance
    prev = max(1, prev)
    next = min(latest_block, next)
    if prev == next:
        counter=0
        tolerance=0
        return prev

    t0, t1 = T(prev), T(next)

    blocktime = (t1 - t0) / (next-prev)
    k = (timestamp - t0) / (t1-t0)
    block_predicted = int(prev + k * (next - prev))

    time_predicted = T(block_predicted)

    blocks_diff = int((timestamp - time_predicted) / blocktime)
    adjustment = block_predicted + blocks_diff

    r = abs(blocks_diff)
    #tolerance
    if r <= tolerance:
        return(adjustment)
    counter +=1
    if counter > 10:
        tolerance+=1
    return block_by_time(timestamp, adjustment - r, adjustment + r)


# Returns block closest to given TTD value
def block_by_ttd(ttd, prev, next):
    prev = max(1, prev)
    next = min(latest_block, next)

    if prev == next:
        return prev

    t0, t1 = TTD(prev), TTD(next)

    difficulty = (t1 - t0) / (next-prev)
    k = (ttd - t0) / (t1-t0)

    block_predicted = int(prev + k * (next - prev))

    ttd_predicted = TTD(block_predicted)

    blocks_diff = int((ttd - ttd_predicted) / difficulty)
    adjustment = block_predicted + blocks_diff

    r = abs(blocks_diff)
    
    if r <= 1:
        return(adjustment)

    return block_by_ttd(ttd, adjustment - r, adjustment + r)

# Updates data set with latest blocks
def update(blockn, step, row):
    ts=web3.eth.getBlock(blockn).timestamp
    latest_ts = web3.eth.get_block('latest')['timestamp']

    points = (latest_ts-ts) / step
    block_step=step/13
    i=row

    while i < int(row+points):
        ttd=int(web3.eth.getBlock(blockn).totalDifficulty / 100000 )
        difficulty=web3.eth.getBlock(blockn).difficulty
        ts=int(web3.eth.getBlock(blockn).timestamp)
#        print(row/i*100)
        print('Updating data at block', blockn)
        data = {
        'Row': [i],
        'BlockNumber': [blockn],
        'TTD': [ttd],
        'Difficulty': [difficulty],
        'UnixTimestamp': [ts]
        }

        df = pd.DataFrame(data)
        df.to_csv('result.csv', mode='a', index=False, header=False)

        next = min(latest_block, int(blockn+block_step))

        blockn=block_by_time((ts+step), next, latest_block)
        i+=1

# Creates polynomial equation following collected data
def construct_polynom():

    csv = pd.read_csv('./result.csv')
    data = csv[['Row', 'BlockNumber', 'TTD', 'UnixTimestamp']]
    x = data['Row']
    y = data['TTD']
    t = data['UnixTimestamp']
    l=int(len(y))-1

    predict = np.polyfit(t, y, degree)
    predict_time = np.polyfit(x, t, degree)
    
    #mean squared error
    err=[]
    for i in range(l):
        diff=abs(np.polyval(predict,int(t[i]))-y[i])
        err.append(diff**2)
    MSE=np.average(err)
    #print(MSE, "MSE1")

    ##Training set
    train_size=int(l*0.75)
    t_train=t[0:train_size]
    y_train=y[0:train_size]

    
    predictm = np.polyfit(t_train, y_train, degree)
    errp=[]
    errm=[]
    i=0
    for i in range(l+1):
        diff=abs(np.polyval(predict,int(t[i]))-y[i])
        errp.append(diff+y[i])
        errm.append(y[i]-diff)

    predictp=np.polyfit(t,errp,degree)
    predictm=np.polyfit(t,errm,degree)
    MSE=np.average(err)

    '''
    #Print the equation
    i=0
    while i <= degree:
        if i == degree:
            print("(%.10f)"%(predict[i]))
            break
        print("(%.10f*x^%d)+"%(predict[i],degree-i,),end="")
        i+=1
        '''
    if args['ttd']:
        target_ttd = int(args['ttd']) / 100000
        estimate_ttd(target_ttd, predict, predict_time, predictp, predictm)

def estimate_hashrate(data):

    d=[]
    t = data['UnixTimestamp']
    y = data['TTD']
    current_ttd = web3.eth.get_block('latest')['totalDifficulty']
    target=int(args['ttd']) 
    time_now=web3.eth.get_block('latest')['timestamp']
    time_target=int(args['time'])


    i=1
    while i < l:
        i+=1
        d.append(int ((y[i]-y[i-1]) / (t[i]-t[i-1]))/10000)

    conv=np.vectorize(dt.datetime.fromtimestamp) 
    dates=conv(t)

    ax=plt.gca()
    ax.clear() 

    plt.subplots_adjust(bottom=0.2)
    plt.xticks( rotation=25 )
    xfmt = md.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(xfmt)
    plt.title("Sepolia hashrate")
    ax.set_ylabel('GH/s')
    plt.scatter(dates[1:l], d) 
    plt.plot(dates[1:l], d) 

    plt.savefig('hashrate.png')
    
    n=int(len(y))-3
    ttd_diff_avg=int(np.average(np.diff(y[n:])))
    time_diff_avg=int(np.average(np.diff(t[n:])))

    hashrate=(ttd_diff_avg*100000/time_diff_avg/1000000000)
    print("Current hashrate: %.2f" % hashrate, "GH/s <p></p>")
    hash_target=((target-current_ttd)/(time_target-time_now)/1000000000)
    print("To achieve TTD", target, "at", time.ctime(time_target),"UTC, around %.2f GH/s in the network is needed as of now." % hash_target)

def draw_chart(target_t, target_y, y, t, predict_err, predictm):
    c = np.poly1d(predict_err)
    h = np.poly1d(predictm)

    p = np.poly1d(np.polyfit(t, y, degree))
    ds=np.copy(t)
    time_diff_avg=int(np.average(np.diff(t)))
    i=(target_t-t[int(len(t)-1)])/time_diff_avg + int(len(t))/10
    j=int(len(t))
    while i > 0:
        t[j]=t[j-1]+time_diff_avg
        j+=1
        i-=1
    
    conv=np.vectorize(dt.datetime.fromtimestamp) 
    dates=conv(t)
    ds=conv(ds)
    target_t=conv(target_t)
    target_sep=conv(int(args['time']))
    
    plt.subplots_adjust(bottom=0.2)
    plt.xticks( rotation=25 )
    
    ax=plt.gca()
    xfmt = md.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(xfmt)
    plt.title("Sepolia TTD")
    plt.plot(dates,p(t), color='red', linestyle='dashed')
    plt.plot(dates,c(t), color='purple', linestyle='dashed')
    plt.plot(dates,h(t), color='purple', linestyle='dashed')

    plt.scatter(ds, y) 
    plt.plot(target_t,target_y/100000,'ro', color='red') 
    plt.plot(target_sep,target_y/100000,'ro', color='green') 

    plt.savefig('chart.png')
   # plt.show()

#Returns estimated time of given TTD value
def estimate_ttd(target, polynom_ttd, polynom_time, predict_err, predict_m):
    
    appoint=np.copy(polynom_ttd)
    appoint[-1] -= target
    point=int(max(np.roots(appoint)))


    appoint=np.copy(predict_err)
    appoint[-1] -= target
    point1=int(max(np.roots(appoint)))


    appoint=np.copy(predict_m)
    appoint[-1] -= target
    point2=int(max(np.roots(appoint)))

    n=int(len(y))-10
    ttd_diff_avg=int(np.average(np.diff(y[n:])))
    time_diff_avg=int(np.average(np.diff(t[n:])))
    currenttd = web3.eth.get_block('latest')['totalDifficulty']

    draw_chart(point,int(args['ttd']),y,t, predict_err, predict_m)
    
    if point <= 0 or currenttd > target * 100000:
        print("TTD of", target, "was achieved at block", block_by_ttd(target*100000, 1, latest_block))
    else:
        timeleft=(int(target)*100000-currenttd)/(ttd_diff_avg*100000)*time_diff_avg 
        if timeleft < 10000:
          print("Current TTD is", currenttd,"and using latest data lineary, around", datetime.timedelta(seconds =int(timeleft)), "is left to achieve the target.")
      #  print("Total Terminal Difficulty of", int(args['ttd']), "is expected around", time.ctime(point), ", i.e. between", time.ctime(point1),"and", time.ctime(point2), "<p></p>")
        estimate_hashrate(data)


#Returns estimated TTD value at given timestamp
def estimate_time(target, polynom_ttd ):
    
    point=target
    draw_chart(int(args['time']), int(np.polyval(polynom_ttd,point)*100000),y,t)
    
    #some edgecases to handle here, crazy timestamp values will run into error
    if point <= l:
        print("Time of", target, "was achieved at", time.ctime(t[point]), "block", block_by_ttd(target*100000, int(b[point]), int(b[point]+1)))
    else:
     
#        print("Total Terminal Difficulty at time", time.ctime(target), "UTC is expected around value", int(np.polyval(polynom_ttd,point))*100000)
        return int(np.polyval(polynom_ttd,point))*100000


ap = argparse.ArgumentParser()
ap.add_argument("--ttd", required=False,
   help="Total terminal difficulty value to predict")

ap.add_argument("--time", required=False,
   help="Timestamp to predict")

args = vars(ap.parse_args())


if os.path.exists('result.csv'):
    csv = pd.read_csv('./result.csv')
    data = csv[['Row', 'BlockNumber', 'TTD', 'Difficulty', 'UnixTimestamp']]
    x = data['Row']
    y = data['TTD']
    t = data['UnixTimestamp']
    b = data['BlockNumber']
    d = data ['Difficulty']
    l=int(len(y))-1

    latest = web3.eth.get_block('latest')['timestamp']
    ts_now=int(time.time())

    if ( ts_now - t[l]) > granuality:
        next_ts=(t[l]+granuality)
        start_block=block_by_time(next_ts, int(b[l]), latest_block)
        update(start_block, granuality, l+2)
        
else: 
    with open('result.csv', 'a') as file:
        file.write('Row,BlockNumber,TTD,Difficulty,UnixTimestamp\n')

    update(start, granuality, 1)


construct_polynom()