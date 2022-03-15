from web3 import Web3
from web3.types import BlockData
import sys

web3 = Web3(Web3.HTTPProvider("https://bordel.xyz"))
#web3 = Web3(Web3.IPCProvider("~/.ethereum/geth.ipc"))
#web3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))


T = lambda blockn: web3.eth.getBlock(blockn).timestamp

latest = web3.eth.get_block('latest')['number']

def target_block(timestamp, prev=1, next=latest):
    prev = max(1, prev)
    next = min(latest, next)

    if prev == next:
        print(prev)
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
    if r <= 1:
        print(adjustment)
        return(adjustment)

    return target_block(timestamp, adjustment - r, adjustment + r)

ts, target = int(sys.argv[1]), int(sys.argv[2])
block = target_block(ts, target)

