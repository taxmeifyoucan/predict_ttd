# Calculates and visualizes how much hashrate is needed to achieve given TTD during September 
# Run with TTD to estimate in argument, like so:
# python3 hashrate_ttd.py --ttd  58750000000000000000000

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
from web3 import Web3
import datetime as dt
import argparse 


#Connect to web3 provider
#web3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))

T = lambda blockn: web3.eth.getBlock(blockn).timestamp
TTD = lambda blockn: web3.eth.getBlock(blockn).totalDifficulty
latest_block = web3.eth.get_block('latest')['number']

def estimate_hashrate(target):

    current_ttd = web3.eth.get_block('latest')['totalDifficulty']
    time_now=web3.eth.get_block('latest')['timestamp']
    hashrate=((TTD("latest")-TTD(latest_block-6450))/(T("latest")-T(latest_block-6450))/1000000000000) #hashrate in roughly past day
    time_targets=[1662033600, 1662638400, 1663243200, 1663848000, 1664539200, 1665144000]

    #September 
    start=1661983200
    end=1664748000

    t=[]
    t.append(start)
    i=0
    while t[i] < end:
        i+=1
        t.append(t[i-1]+43200)
    
    i=0
    h=[]
    p=[]

    for timet in t:
        h.append(((target-current_ttd)/(timet-time_now)/1000000000000))

    for hash_t in h:
        p.append(((hashrate-hash_t)/hashrate)*(-100))

    conv=np.vectorize(dt.datetime.fromtimestamp) 
    dates=conv(t)
    
    ax=plt.gca()
    plt.subplots_adjust(bottom=0.2)
    plt.xticks( rotation=25 )
    xfmt = md.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(xfmt)
    plt.title("Hashrate change")
    ax.set_ylabel('Δ %')
    plt.plot(dates, p)
    plt.axhline(y = 0, color = 'r', linestyle = 'dashed')
    plt.savefig('percent_delta.png')
    ax.grid(True)
    plt.show()
    plt.clf()
    ax.clear()

    ax=plt.gca()
    plt.subplots_adjust(bottom=0.2)
    plt.xticks( rotation=25 )
    xfmt = md.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(xfmt)
    plt.title("Hashrate to achieve TTD")
    ax.set_ylabel('TH/s')
    plt.plot(dates, h)
    plt.axhline(y = hashrate, color = 'r', linestyle = 'dashed')
    ax.grid(True)
    plt.savefig('hashrate_delta.png')
    plt.show()

    for time_target in time_targets:
        hash_target=((target-current_ttd)/(time_target-time_now)/1000000000000)
        delta=((hashrate-hash_target)/hashrate)*100
        print("To achieve TTD", target, "at", dt.datetime.utcfromtimestamp(time_target).strftime("%a %b %d %H:%M %Y"),"UTC, around %.1f TH/s in the network is needed as of now." % hash_target)
        print("That is around", int(delta), "% change from current hashrate")

ap = argparse.ArgumentParser()
ap.add_argument("--ttd", required=True,
   help="TTD value to estimate")

args = vars(ap.parse_args())

target = int(args['ttd']) 

estimate_hashrate(target)