import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from web3 import Web3
import argparse 
import time
import os.path
import warnings

warnings.filterwarnings('ignore')

#Choose web3 provider first, IPC is recommended 
#web3 = Web3(Web3.IPCProvider("~/.ethereum/geth.ipc"))
#web3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))

#If result.csv is not present, data will be crawled based on these parameters:
start=14603000 #First block to start with 
granuality=1800 #Step in seconds
degree=2 #Degree of polynomials

T = lambda blockn: web3.eth.getBlock(blockn).timestamp
TTD = lambda blockn: web3.eth.getBlock(blockn).totalDifficulty
latest_block = web3.eth.get_block('latest')['number']


# Binary search which finds block closest to given timestamp
def block_by_time(timestamp, prev, next):
    prev = max(1, prev)
    next = min(latest_block, next)

    if prev == next:
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
    if r <= 2:
        return(adjustment)

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
        blockn=block_by_time((ts+step), int(blockn+block_step), int(latest_block))
        i+=1

# Creates polynomial equation following collected data
def construct_polynom():

    coeff_ttd = np.polyfit(x, y, degree)
    coeff_time = np.polyfit(x, t, degree)
    
    plt.scatter(x, y) 
    p = np.poly1d(coeff_ttd)
    plt.plot(x,p(x),"r--")
    plt.savefig('chart.png')

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
    if args['ttd']:
        target_ttd = int(args['ttd']) / 100000
        estimate_ttd(target_ttd, coeff_ttd, coeff_time)
    if args['time']:
        estimate_time(int(args['time']), coeff_ttd, coeff_time)

#Returns estimated time of given TTD value
def estimate_ttd(target, polynom_ttd, polynom_time):
    
    #Find x for given y
    appoint=np.copy(polynom_ttd)
    appoint[-1] -= target
    point=int((np.roots(appoint)[degree-1]))

    #Calculated averages from data
    ttd_diff_avg=int(np.average(np.diff(y)))
    time_diff_avg=int(np.average(np.diff(t)))
    current_ttd = web3.eth.get_block('latest')['totalDifficulty']
    
    if point <= 0 or current_ttd > target * 100000:
        print("TTD of", target, "was achieved at block", block_by_ttd(target*100000, 1, latest_block))
    else:
        timeleft=(int(target)*100000-current_ttd)/(ttd_diff_avg*100000)*time_diff_avg
        if timeleft < 86400:
            print(time.strftime("Around %Hh%Mm%Ss left \n", time.gmtime(timeleft)))
     
        deviation=(abs(((point - l)*time_diff_avg+t[l])-np.polyval(polynom_time,point)))
        mid=(((point - l)*time_diff_avg+t[l])+np.polyval(polynom_time,point))/2
        naive=abs((point-l)*time_diff_avg+t[l] - mid)
        print("Total Terminal Difficulty of", target, "is expected around", time.ctime(np.polyval(polynom_time,point)), ", i.e. between",time.ctime(mid-deviation),"and",time.ctime(mid+deviation))

#Returns estimated TTD value at given timestamp
def estimate_time(target, polynom_ttd, polynom_time):

    appoint=np.copy(polynom_time)
    appoint[-1] -= target
    point=int((np.roots(appoint)[degree-1]))
    step=int(np.average(np.diff(y)))

    #some edgecases to handle here, crazy timestamp values will run into error
    if point <= l:
        print("Time of", target, "was achieved at", time.ctime(t[point]), "block", block_by_ttd(target*100000, int(b[point]), int(b[point]+30)))
    else:
        deviation=(abs(((point - l)*step+y[l])-np.polyval(polynom_ttd,point)))
        mid=(((point - l)*step+y[l])+np.polyval(polynom_ttd,point))/2
     
        print("Total Terminal Difficulty at time", time.ctime(target), " is expected around value", int(np.polyval(polynom_ttd,point)*100000))


ap = argparse.ArgumentParser()
ap.add_argument("--ttd", required=False,
   help="Total terminal difficulty value to predict")

ap.add_argument("--time", required=False,
   help="Timestamp to predict")

args = vars(ap.parse_args())


if os.path.exists('result.csv'):
    csv = pd.read_csv('./result.csv')
    data = csv[['Row', 'BlockNumber', 'TTD', 'Difficulty', 'UnixTimestamp']]
    b = data['BlockNumber']
    t = data['UnixTimestamp']
    l=int(len(b))-1

    latest = web3.eth.get_block('latest')['timestamp']
    ts_now=int(time.time())

    if ( ts_now - t[l]) > granuality:
        next_ts=(t[l]+granuality)
        start_block=block_by_time(next_ts, int(b[l]), latest_block)
        update(start_block, granuality, l+2)
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
else: 
    with open('result.csv', 'a') as file:
        file.write('Row,BlockNumber,TTD,Difficulty,UnixTimestamp\n')

    update(start, granuality, 1)

    construct_polynom()



