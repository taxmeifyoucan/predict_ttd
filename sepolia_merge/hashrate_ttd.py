  
import numpy as np
from web3 import Web3
import time
import datetime as dt
  
#Choose web3 provider, use IPC especially for a lot of blocks in the past
web3 = Web3(Web3.IPCProvider("/home/mario/.ethereum/sepolia/geth.ipc"))

T = lambda blockn: web3.eth.getBlock(blockn).timestamp
TTD = lambda blockn: web3.eth.getBlock(blockn).totalDifficulty
D = lambda blockn: web3.eth.getBlock(blockn).difficulty

#Choose values to predict
target_ttd=17000000000000000
target_time=1656936000 #4 july
past_blocks=100 #n of blocks in the past to calculate with
latest_block = web3.eth.get_block('latest')['number']

def hashrate():

    latest_block = web3.eth.get_block('latest')['number']
    #Using latest data
    latest_ttd=TTD(latest_block)
    latest_time=T(latest_block)

    ttds=[]
    times=[]
    diffs=[]
    hashrates=[]


    i=past_blocks
    while i > 0:
        ttds.append(TTD(latest_block-i))
        times.append(T(latest_block-i))
        diffs.append(D(latest_block-i))
        i-=5

    time_avg=np.average(np.diff(times))
    ttd_avg=np.average(np.diff(ttds))

    i=0
    while i < len(ttds)-1:
        i+=1
        hashrates.append(int((ttds[i]-ttds[i-1]) / (times[i]-times[i-1])))
        

    avg_hashrate=(ttd_avg/time_avg/1000000000)
    print("Recent hashrate in past", past_blocks,"blocks is %f GH/s" % avg_hashrate)
    timeleft=int(((target_ttd-latest_ttd)/(ttd_avg))*time_avg)
    if timeleft < 7200:
        print("Around", dt.timedelta(seconds =int(timeleft)), "left")    
    return timeleft

timeleft=hashrate()

latest = web3.eth.get_block('latest')['number']

while timeleft > 0:
    latest = web3.eth.get_block('latest')['number']
    if latest > latest_block:
        latest_block=latest
        print("Block", latest)
        timeleft=hashrate()
    time.sleep(0.5)
