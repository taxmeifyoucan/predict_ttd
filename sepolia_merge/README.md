## Sepolia Merge prediction
To get estimation of hashrate needed to achieve target on time, run the script with the TTD target and time target values.
```
python3 wensepolia.py --ttd 17000000000000000 --time 1657119600 
``` 
Prediction when mining began. Let's move the red dot to the green one!
![](./chart.png)

![](./hashrate.png)

`hashrate_ttd.py` provides live feed of current blocks and how they affect the hashrate. It pulls all data live so it's recomended to connect with IPC. You can change `past_blocks` value to calculate hashrate from how many past blocks you wish. 
