# Predicting TTD 

This repository contains tooling and data for estimating TTD values in Ethereum network. The main purpose is to craft TTD value and track it before The Merge. Current results are published at https://bordel.wtf.

Terminal Total Difficulty is an accumulated sum of miner work done at each block. [The Merge](https://ethereum.org/en/upgrades/merge/) will happen when a certain value TTD value is reached. This tool makes it easier to predict when network reaches given TTD. It was used to calculated TTDs for Merges in various networks including mainnet. Data can be found in corresponding directories. 

Program for predicting the value collects data on accumulated difficulty with precise points in time and naively estimates future TTD using linear/polynomial regression. Keep in mind that further in the future your prediction is, the bigger error there will be. 

Deeper explanation of technical strategy of TD prediction used by this tool can be found in [this post](https://ethresear.ch/t/predicting-ttd-on-ethereum/12742). 

## Usage 

The repository includes various python scripts. Each needs to connet to some Ethereum network and requires web3 provider, an RPC endpoint. Make sure to set yours on top of file you are going to run. 

To use these scripts, you need Python3.7+ and pip to install requirements:

```
pip3 install -r requirements.txt
```

### Predicting TTD

The main prediction tools is `wenmerge.py`.

When there is no present `result.csv` file with collected data from the network, it starts to collect data 4 weeks in the past with 2 hours step. You can set values for starting time and step. To collect your own data set. Local node IPC is recommended for this step because the initial data collection can take a while. 

Some already created data for prediction can be found in corresponding directories. Data are updated when you run the script to create the latest prediction. 

To just collect the data, you can run the script without any flags. To define values to predict, use arguments as in example below. `--ttd` creates a prediction of when is given total difficulty going to be achieved and `--time` estimates total difficulty value at given timestamp. If both flags are supplied, script also calculates current hashrate and estimates how much hashrate is needed to achieve given TTD at given timestamp. 

```
python3 wenmerge.py --ttd 58750000000000000000000 --time 1663243200
```

Here is an example of terminal output: 

```
Updating data at block 15385601

Terminal Total Difficulty of 58750000000000000000000 is expected around Thu Sep 15 04:09:42 2022 , i.e. between Thu Sep 15 05:16:44 2022 and Thu Sep 15 03:02:47 2022
Terminal Total Difficulty at time Thu Sep 15 14:00:00 2022 UTC is expected around value 58781255664353442529280
Current daily hashrate: 864.5 TH/s
To achieve TTD 58750000000000000000000 at Thu Sep 15 14:00:00 2022 UTC, around 870.0 TH/s in the network is needed as of now.
```

It also plots charts of predicted total difficulty and current hashrate. To examine them closely, you can uncomment `plt.show()` and they will be displayed when the program is run.

If the prediction gets stuck, try using a lower degree of polynomial or collecting a new dataset. 

Predicted values can also be served as an API and displayed on a web page, check [api](/api) for details. 

### Estimating hashrate

Two other scripts you can find here are dedicated to estimating hashrate values. 

`ttd_hashrate.py` creates an estimation of how much hashrate is needed to achieve a given TTD during a given time span and specific dates (predefined to September 2022). It calculates the hashrate in TH/s needed to reach the total difficulty value in a given time and compares it to the current hashrate. Charts visualizing the hashrate and the percent change are also printed. You can easily see when the current hashrate reaches the total difficulty and when it would be reached if it drops. 

Use the TTD value as an argument.

```
python3 hashrate_ttd.py --ttd  58750000000000000000000

To achieve TTD 58750000000000000000000 at Thu Sep 15 12:00:00 2022 UTC, around 869.93 TH/s in the network is needed as of now.
That is around 0 % change from current hashrate
To achieve TTD 58750000000000000000000 at Thu Sep 22 12:00:00 2022 UTC, around 676.97 TH/s in the network is needed as of now.
That is around 22 % change from current hashrate

```

`latest_hashrate.py` calculates hashrate over past n blocks, by default 5000. It runs continuously and updates the hashrate with the latest blocks. This is especially useful with a high volatility hashrate in testnets. 
