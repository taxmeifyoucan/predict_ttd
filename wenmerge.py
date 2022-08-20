import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md
from web3 import Web3
import datetime as dt
import time
import os
import warnings
import argparse 
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings('ignore')

#Choose web3 provider first, IPC is recommended 
#web3 = Web3(Web3.IPCProvider("~/.ethereum/geth.ipc"))
#web3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))

T = lambda blockn: web3.eth.getBlock(blockn).timestamp
TTD = lambda blockn: web3.eth.getBlock(blockn).totalDifficulty
latest_block = web3.eth.get_block('latest')['number']

#If result.csv is not present, data will be collected since this start block and with given granuality 
begin=2419200 #First data point starts this long before, 2419200 seconds = 4 weeks ago
granuality=7200 #Step between data points
degree=2 #Degree of polynomials
tolerance = counter = 1

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

    if r <= tolerance:
        return(adjustment)

    counter +=1
    if counter > 10:
        tolerance+=1
    return block_by_time(timestamp, adjustment - r, adjustment + r)


# Returns block closest to given TTD value
def block_by_ttd(ttd, prev, next):
    global counter
    global tolerance

    prev = max(1, prev)
    prev=min(prev, latest_block)
    next = min(latest_block, next)

    if prev == next:
        counter=0
        tolerance=0  
        return prev

    t0, t1 = TTD(prev), TTD(next)
    difficulty = (t1 - t0) / (next-prev)
    k = (ttd - t0) / (t1-t0)
    block_predicted = int(prev + k * (next - prev))
    ttd_predicted = TTD(block_predicted)
    blocks_diff = int((ttd - ttd_predicted) / difficulty)
    adjustment = block_predicted + blocks_diff
    r = abs(blocks_diff)

    if r <= tolerance:
        return(adjustment)

    counter +=1
    if counter > 10:
        tolerance+=1

    return block_by_ttd(ttd, adjustment - r, adjustment + r)

def draw_chart(target_t, target_y, poly_h, poly_l):
   
    ph = np.poly1d(poly_h)
    pl = np.poly1d(poly_l)
    p = np.poly1d(np.polyfit(t, y, degree))

    tt = t[:]
    timestamps=np.copy(tt)
    time_diff_avg=int(np.average(np.diff(tt)))
    i=(target_t-tt[int(len(tt)-1)])/time_diff_avg + int(len(tt))/10 
    j=int(len(tt))

    while i > 0:
        tt[j]=tt[j-1]+time_diff_avg
        j+=1
        i-=1

    conv=np.vectorize(dt.datetime.fromtimestamp) 
    long_dates=conv(tt)
    short_dates=conv(timestamps)
    target_t=conv(target_t)
    
    plt.subplots_adjust(bottom=0.2)
    plt.xticks( rotation=25 )
    ax=plt.gca()
    ax.minorticks_on()
    xfmt = md.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(xfmt)
    plt.title("Total Terminal Difficulty")
    ax.set_ylabel('Accumulated difficulty')
    ax.grid(True)
    plt.plot(long_dates,p(tt), color='red', linestyle='dashed')
    plt.plot(long_dates,ph(tt), color='purple', linestyle='dashed')
    plt.plot(long_dates,pl(tt), color='purple', linestyle='dashed')
    plt.plot(short_dates, y, linewidth=3.5, alpha=0.8) 
    plt.plot(target_t,target_y/10000,'ro', color='green') 
    
    plt.savefig('chart.png', transparent=False, dpi=110)
    #plt.show()

# Updates data set with latest blocks
def update(blockn, step):
    ts=web3.eth.getBlock(blockn).timestamp
    latest_ts = web3.eth.get_block('latest')['timestamp']

    block_step=int(np.average(np.diff(b)))

    while blockn < latest_block:
        ttd=int(web3.eth.getBlock(blockn).totalDifficulty / 10000 )
        difficulty=web3.eth.getBlock(blockn).difficulty
        ts=int(web3.eth.getBlock(blockn).timestamp)

        print('Updating data at block', blockn)
        data = {
        'BlockNumber': [blockn],
        'TTD': [ttd],
        'Difficulty': [difficulty],
        'UnixTimestamp': [ts]
        }

        df = pd.DataFrame(data)
        df.to_csv('result.csv', mode='a', index=False, header=False)

        next = min(latest_block, int(blockn+block_step))

        blockn=block_by_time((ts+step), next, latest_block)

def estimate_hashrate(target, target_time):

    d=[]
    l=int(len(y))-1

    current_ttd = web3.eth.get_block('latest')['totalDifficulty']
    time_now=web3.eth.get_block('latest')['timestamp']
    time_targets=[ target_time ]
    i=1
    while i < l:
        i+=1
        d.append(int ((y[i]-y[i-1]) / (t[i]-t[i-1]))/1000000000000)
    hash_coeff = np.poly1d(np.polyfit(t[1:l], d, 10))

    conv=np.vectorize(dt.datetime.fromtimestamp) 
    dates=conv(t)

    ax=plt.gca()
    ax.clear() 

    plt.subplots_adjust(bottom=0.2)
    plt.xticks( rotation=25 )
    xfmt = md.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(xfmt)
    plt.title("Hashrate")
    ax.set_ylabel('TH/s')
    plt.scatter(dates[1:l], d) 
    plt.plot(dates[1:l], d) 
    plt.plot(dates[1:l],hash_coeff(t[1:l]), color='red', linestyle='dashed')
    plt.savefig('hashrate.png', transparent=False, dpi=110)
    plt.clf()
    time_avg=int(np.average(np.diff(t)))
    n=int(len(y)-(86400/time_avg))
    ttd_diff_avg=int(np.average(np.diff(y[n:])))
    time_diff_avg=int(np.average(np.diff(t[n:])))

    hashrate=(ttd_diff_avg*10000/time_diff_avg/1000000000000)
    print("Current daily hashrate: %.1f" % hashrate, "TH/s")
    hashrate_projection=time.ctime(int((target-current_ttd)/(ttd_diff_avg*10000/time_diff_avg))+time_now)

    for time_target in time_targets:
        hash_target=((target-current_ttd)/(time_target-time_now)/1000000000000)
        print("To achieve TTD", target, "at", time.ctime(time_target),"UTC, around %.1f TH/s in the network is needed as of now." % hash_target)

    return hashrate, hashrate_projection

def construct_errors(coeff_ttd):

    l=int(len(y))-1

    #Errors 
    #Training split
    train_size=int(l*0.5)
    t_train=t[0:train_size]
    y_train=y[0:train_size]

    coeff_train = np.polyfit(t_train, y_train, degree)

    err_h=[]
    err_l=[]
    mse=[]
    for i in range(l+1):
        diff=abs(np.polyval(coeff_train,int(t[i]))-y[i])
        err_h.append(diff+y[i])
        err_l.append(y[i]-diff)

        diff=abs(np.polyval(coeff_ttd,int(t[i]))-y[i])
        mse.append(diff**2)
    
    coeff_h=np.polyfit(t,err_h,degree)
    coeff_l=np.polyfit(t,err_l,degree)

    #Mean squared error
    MSE=np.average(mse)
    #print("MSE: ", MSE)

    return coeff_h, coeff_l
    
#Returns estimated time of given TTD value
def estimate_ttd(target):

    coeff_ttd = np.polyfit(t, y, degree)
    coeff_h, coeff_l=construct_errors(coeff_ttd)
    ttd_target=target/10000
    
    #Find x for given y
    substitute=np.copy(coeff_ttd)
    substitute[-1] -= ttd_target
    point=int(max(np.roots(substitute)))

    substitute=np.copy(coeff_h)
    substitute[-1] -= ttd_target
    point_high=int(max(np.roots(substitute)))

    substitute=np.copy(coeff_l)
    substitute[-1] -= ttd_target
    point_low=int(min(np.roots(substitute)))

    #Calculated averages from data
    ttd_diff_avg=int(np.average(np.diff(y)))
    time_diff_avg=int(np.average(np.diff(t)))
    current_ttd = web3.eth.get_block('latest')['totalDifficulty']

    draw_chart(point,target,coeff_h, coeff_l)

    if point <= 0 or current_ttd > target:
        print("TTD of", target, "was achieved at block", block_by_ttd(target*10000, 1, latest_block))
    else:
        timeleft=(int(target)*10000-current_ttd)/(ttd_diff_avg*10000)*time_diff_avg
        if timeleft < 259200:
            print("Around", dt.timedelta(seconds =timeleft), "left")

        print("Total Terminal Difficulty of", int(target), "is expected around", time.ctime(point), ", i.e. between", time.ctime(point_low),"and", time.ctime(point_high))

        return time.ctime(point)

#Returns estimated TTD value at given timestamp
def estimate_time(target):

    coeff_ttd = np.polyfit(t, y, degree)
    coeff_h, coeff_l=construct_errors(coeff_ttd)

    td=int(np.polyval(coeff_ttd, target)*10000)

    draw_chart(target, td, coeff_h, coeff_l)

    #some edgecases to handle here, crazy timestamp values will run into error
    if T('latest') > target:
        print("Time of", target, "was achieved at", time.ctime(t[target]), "block", block_by_ttd(target*10000, int(b[1]), int(b[l]+1)))
    else:
        print("Total Terminal Difficulty at time", time.ctime(target), "UTC is expected around value", td)
        return td

ap = argparse.ArgumentParser()
ap.add_argument("--ttd", required=False,
   help="Total terminal difficulty value to predict")

ap.add_argument("--time", required=False,
   help="Timestamp to predict")

args = vars(ap.parse_args())


if os.path.exists('result.csv'):
    csv = pd.read_csv('./result.csv')
    data = csv[['BlockNumber', 'TTD', 'Difficulty', 'UnixTimestamp']]
    b = data['BlockNumber']
    t = data['UnixTimestamp']
    l=int(len(t))-1
    ts_now=int(time.time())
    time_diff=int(np.average(np.diff(t)))

    if ( ts_now - t[l]) > time_diff:
        next_ts=(t[l]+time_diff)
        start_block=block_by_time(next_ts, int(b[l]), latest_block)
        update(start_block, time_diff)
else: 

    with open('result.csv', 'a') as file:
        file.write('BlockNumber,TTD,Difficulty,UnixTimestamp\n')

    start=(block_by_time((T('latest')-begin), 15000000, latest_block)) 
    update(start, granuality)

csv = pd.read_csv('./result.csv')
data = csv[['BlockNumber', 'TTD', 'Difficulty', 'UnixTimestamp']]
y = data['TTD']
d = data ['Difficulty']
b = data['BlockNumber']
t = data['UnixTimestamp']


if args['ttd']:
    target = int(args['ttd'])
    estimate_ttd(target)

if args['time']:
    target = int(args['time'])
    estimate_time(target)

if args['time'] and args['ttd']:
    estimate_hashrate(int(args['ttd']),int(args['time']))

