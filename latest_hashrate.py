import numpy as np
from web3 import Web3
import time
import datetime as dt
  
#Choose web3 provider, use IPC especially for a lot of blocks in the past
web3 = Web3(Web3.HTTPProvider("http://localhost:8545"))
web3 = Web3(Web3.HTTPProvider("https://bordel.xyz/"))

T = lambda blockn: web3.eth.getBlock(blockn).timestamp
TTD = lambda blockn: web3.eth.getBlock(blockn).totalDifficulty
latest_block = web3.eth.get_block('latest')['number']

#Value to 
target_ttd=58750000000000000000000
#n of blocks in the past to calculate with
past_blocks=5000

def hashrate(latest_block):

    latest_ttd=TTD(latest_block)

    ttds=[]
    times=[]
    hashrates=[]

    i=past_blocks
    while i > 0:
        ttds.append(TTD(latest_block-i)/10000)
        times.append(T(latest_block-i))
        i-=int(past_blocks/2)

    time_avg=np.average(np.diff(times))
    ttd_avg=np.average(np.diff(ttds))

    avg_hashrate=(ttd_avg/time_avg/100000000)
    print("Recent hashrate in past", past_blocks,"blocks is %.f TH/s" % avg_hashrate)
    timeleft=int(((target_ttd-latest_ttd)/(ttd_avg))*time_avg)    
    if timeleft < 0:
        print ("TTD achieved!")
    elif timeleft < 43200: 
        print("Around", dt.timedelta(seconds =int(timeleft)), "left") 
    return timeleft

timeleft=hashrate(latest_block)

latest = web3.eth.get_block('latest')['number']

while timeleft > 0:
    latest = web3.eth.get_block('latest')['number']
    if latest > latest_block:
        latest_block=latest
        print("Block", latest)
        timeleft=hashrate(latest_block)
    time.sleep(1)
