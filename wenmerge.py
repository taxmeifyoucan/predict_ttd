import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md
from web3 import Web3
import datetime as dt
import time
import os
import warnings
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings('ignore')
web3 = Web3(Web3.IPCProvider("/home/mario/.ethereum/ropsten/geth.ipc"))

#Choose web3 provider first, IPC is recommended 
#web3 = Web3(Web3.IPCProvider("~/.ethereum/geth.ipc"))
#web3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))

T = lambda blockn: web3.eth.getBlock(blockn).timestamp
TTD = lambda blockn: web3.eth.getBlock(blockn).totalDifficulty
latest_block = web3.eth.get_block('latest')['number']

#If result.csv is not present, data will be crawled based on these parameters:
start=int(latest_block-(50000)) #First block to start with 
granuality=1800 #Step in seconds
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

def draw_chart(target_t, target_y, y, t, poly_h, poly_l):
   
    ph = np.poly1d(poly_h)
    pl = np.poly1d(poly_l)
    p = np.poly1d(np.polyfit(t, y, degree))

    timestamps=np.copy(t)

    time_diff_avg=int(np.average(np.diff(t)))
    i=(target_t-t[int(len(t)-1)])/time_diff_avg + int(len(t))/10 
    j=int(len(t))
    while i > 0:
        t[j]=t[j-1]+time_diff_avg
        j+=1
        i-=1
    
    conv=np.vectorize(dt.datetime.fromtimestamp) 
    long_dates=conv(t)
    short_dates=conv(timestamps)
    target_t=conv(target_t)
    
    plt.subplots_adjust(bottom=0.2)
    plt.xticks( rotation=25 )
    ax=plt.gca()
    xfmt = md.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(xfmt)
    plt.plot(long_dates,p(t), color='red', linestyle='dashed')
    plt.plot(long_dates,ph(t), color='purple', linestyle='dashed')
    plt.plot(long_dates,pl(t), color='purple', linestyle='dashed')

    plt.scatter(short_dates, y) 
    plt.plot(target_t,target_y/100000,'ro', color='green') 
    plt.savefig('chart.png')
   # plt.show()


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

def estimate_hashrate(data):

    d=data['Difficulty']
    t = data['UnixTimestamp']
    y = data['TTD']
    current_ttd = web3.eth.get_block('latest')['totalDifficulty']
    target=43531756765713534
    time_now=web3.eth.get_block('latest')['timestamp']
    time_target=1654706871



    n=1
    ttd_diff_avg=int(np.average(np.diff(y[n:])))
    time_diff_avg=int(np.average(np.diff(t[n:])))

    hashrate=(ttd_diff_avg*100000/time_diff_avg/1000000)
    print(hashrate)
    hash_target=((target-current_ttd)/(time_target-time_now)/1000000)
    print(hash_target)



# Creates polynomial equation following collected data
def construct_polynom(switch):
    csv = pd.read_csv('./result.csv')
    data = csv[['Row', 'BlockNumber', 'TTD', 'Difficulty', 'UnixTimestamp']]
    x = data['Row']
    y = data['TTD']
    t = data['UnixTimestamp']

    coeff_ttd = np.polyfit(t, y, degree)
    coeff_time = np.polyfit(x, t, degree)


    #Errors 
    err_h=[]
    err_l=[]
    err=[]
    for i in range(l+1):
        diff=abs(np.polyval(coeff_ttd,int(t[i]))-y[i])
        err.append(diff**2)
        err_h.append(diff+y[i])
        err_l.append(y[i]-diff)
    
    coeff_h=np.polyfit(t,err_h,degree)
    coeff_l=np.polyfit(t,err_l,degree)

    #Mean squared error
    MSE=np.average(err)
    #print("MSE: ", MSE)

    #Training split
    train_size=int(l*0.75)
    t_train=t[0:train_size]
    y_train=y[0:train_size]

    coeff_train = np.polyfit(t_train, y_train, degree)

    #Print the equation
    '''
    i=0
    while i <= degree:
        if i == degree:
            print("(%.10f)"%(predict[i]))
            break
        print("(%.10f*x^%d)+"%(predict[i],degree-i,),end="")
        i+=1
    '''
    if int(switch) == 1:
        target_ttd = int(os.environ['TTD_TARGET']) / 100000
        return(estimate_ttd(target_ttd, coeff_ttd, coeff_time, x, y, coeff_h, coeff_l ))
    elif 'TIME_TARGET' in os.environ:
        estimate_time(int(os.environ['TIME_TARGET']), coeff_ttd, coeff_h, coeff_l, y)

#Returns estimated time of given TTD value
def estimate_ttd(target, polynom_ttd, polynom_time, x, y, coeff_h, coeff_l):
    
    #Find x for given y

    substitute=np.copy(polynom_ttd)
    substitute[-1] -= target
    point=int(max(np.roots(substitute)))


    substitute=np.copy(coeff_h)
    substitute[-1] -= target
    point_high=int(max(np.roots(substitute)))


    substitute=np.copy(coeff_l)
    substitute[-1] -= target
    point_low=int(max(np.roots(substitute)))


    #Calculated averages from data
    ttd_diff_avg=int(np.average(np.diff(y)))
    time_diff_avg=int(np.average(np.diff(t)))
    current_ttd = web3.eth.get_block('latest')['totalDifficulty']

    draw_chart(point,int(os.environ['TTD_TARGET']),y,t,coeff_h, coeff_l)

    if point <= 0 or current_ttd > target * 100000:
        print("TTD of", target, "was achieved at block", block_by_ttd(target*100000, 1, latest_block))
    else:
        timeleft=(int(target)*100000-current_ttd)/(ttd_diff_avg*100000)*time_diff_avg
        if timeleft < 259200:
            print("Around", dt.timedelta(seconds =timeleft), "left")

        print("Total Terminal Difficulty of", int(os.environ['TTD_TARGET']), "is expected around", time.ctime(point), ", i.e. between", time.ctime(point_low),"and", time.ctime(point_high))

        return time.ctime(point)

#Returns estimated TTD value at given timestamp
def estimate_time(target, polynom_ttd, coeff_h, coeff_l, y):

    point=target

    draw_chart(int(os.environ['TIME_TARGET']), int(np.polyval(polynom_ttd,point)*100000),y,t,coeff_h, coeff_l)

    #some edgecases to handle here, crazy timestamp values will run into error
    if point <= l:
        print("Time of", target, "was achieved at", time.ctime(t[point]), "block", block_by_ttd(target*100000, int(b[point]), int(b[point]+1)))
    else:
     
        print("Total Terminal Difficulty at time", time.ctime(target), "UTC is expected around value", int(np.polyval(polynom_ttd,point)))


if os.path.exists('result.csv'):
    csv = pd.read_csv('./result.csv')
    data = csv[['Row', 'BlockNumber', 'TTD', 'Difficulty', 'UnixTimestamp']]
    b = data['BlockNumber']
    t = data['UnixTimestamp']
    l=int(len(b))-1
    ts_now=int(time.time())

    if ( ts_now - t[l]) > granuality:
        next_ts=(t[l]+granuality)
        start_block=block_by_time(next_ts, int(b[l]), latest_block)
        estimate_hashrate(data)

        update(start_block, granuality, l+2)
    '''

        csv = pd.read_csv('./result.csv')
        data = csv[['Row', 'BlockNumber', 'TTD', 'Difficulty', 'UnixTimestamp']]
        x = data['Row']
        y = data['TTD']
        d = data ['Difficulty']
        b = data['BlockNumber']
        t = data['UnixTimestamp']
        construct_polynom()
    else:
        csv = pd.read_csv('./result.csv')
        data = csv[['Row', 'BlockNumber', 'TTD', 'Difficulty', 'UnixTimestamp']]
        x = data['Row']
        y = data['TTD']
        d = data ['Difficulty']
        b = data['BlockNumber']
        t = data['UnixTimestamp']
        construct_polynom()
    '''
else: 
    with open('result.csv', 'a') as file:
        file.write('Row,BlockNumber,TTD,Difficulty,UnixTimestamp\n')

    update(start, granuality, 1)




