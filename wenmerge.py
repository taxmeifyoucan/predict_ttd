import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from web3 import Web3
import sys
import time
import os.path

#web3 = Web3(Web3.IPCProvider("~/.ethereum/geth.ipc"))
#web3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))

T = lambda blockn: web3.eth.getBlock(blockn).timestamp
latest = web3.eth.get_block('latest')['number']

#Define starting block if result.csv is not created yet
start=14322444
granuality=3600

# Returns block closest to given timestamp
def block_by_time(timestamp, prev=1, next=latest):
    prev = max(1, prev)
    next = min(latest, next)

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


#Updates data with latest blocks
def update(blockn, step, row):
    ts=web3.eth.getBlock(blockn).timestamp
    latest_ts = web3.eth.get_block('latest')['timestamp']

    points = (latest_ts-ts) / step
    block_step=step/13
    i=row

    while i < row+points:
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
        blockn=block_by_time((ts+step), int(blockn+block_step), int(blockn+block_step+10))
        print (i, row+points)
        i+=1

#Creates polynomial equation following collected data
def construct_polynom(target):

    csv = pd.read_csv('./result.csv')
    data = csv[['Row', 'TTD', 'UnixTimestamp']]
    x = data['Row']
    y = data['TTD']
    t = data['UnixTimestamp']
    l=int(len(y))-1

    degree=3
    step=np.average(np.diff(t))

    predict = np.polyfit(x, y, degree)
    predict_time = np.polyfit(x, t, degree)

    plt.scatter(x, y) 
    p = np.poly1d(predict)
    plt.plot(x,p(x),"r--")
    plt.savefig('chart.png')

    #Print the equation
    i=0
    while i <= degree:
        if i == degree:
            print("(%.10f)"%(predict[i]))
            break
        print("(%.10f*x^%d)+"%(predict[i],degree-i,),end="")
        i+=1
    
    estimate_ttd(target, predict, predict_time, step)

#Returns estimated time of given ttd value
def estimate_ttd(target, polynom_ttd, polynom_time, step):
    degree=3
    appoint=np.copy(polynom_ttd)
    appoint[-1] -= target
    point=int((np.roots(appoint)[degree-1]))

    if point <= l:
        print(time.ctime(t[point]))
    else:
        deviation=(abs(((point - l)*step+t[l])-np.polyval(polynom_time,point)))
        mid=(((point - l)*step+t[l])+np.polyval(polynom_time,point))/2
        print("TTD of", target, " is expected between",time.ctime(mid-deviation),"and",time.ctime(mid+deviation))

target = int(sys.argv[1])

if os.path.exists('result.csv'):
    csv = pd.read_csv('./result.csv')
    data = csv[['Row', 'BlockNumber', 'TTD', 'UnixTimestamp']]
    x = data['Row']
    y = data['TTD']
    t = data['UnixTimestamp']
    b = data['BlockNumber']
    l=int(len(y))-1

    latest = web3.eth.get_block('latest')['timestamp']
    ts_now=int(time.time())

    if ( ts_now - t[l]) > granuality:
        next_ts=(t[l]+granuality)
        start_block=block_by_time(next_ts, int(b[l]), int(b[l]+300))
        update(start_block, granuality, l+2)
        construct_polynom(target)
    else:
        construct_polynom(target)
else: 
    with open('result.csv', 'a') as file:
        file.write('Row,BlockNumber,TTD,Difficulty,UnixTimestamp\n')

    update(start, granuality, 1)

