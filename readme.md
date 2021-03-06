# Predicting TTD 

Total Terminal Difficulty is an accumulated sum of miner work done at each block. [The Merge](https://ethereum.org/en/upgrades/merge/) will happen with certain value of TTD. This tool makes it easier to predict when network reaches certain TTD. 

Checkout article on TTD prediction to get a better understanding of this repo: https://notes.ethereum.org/@MarioHavel/predicting_ttd

Still WIP, precision might be improved, edge cases handled and frontend for presenting prediction to wider community can be created based on this. Program collects data on accumulated difficulty with precise points in time and naively estimates future TTD with linear regression. Keep in mind that further in the future your prediction is, the bigger error there will be. 

## Usage

You need Python3 and pip, first install requirements:
```
pip3 install -r requirements.txt
```

Before running the script, set your web3 provider on the top of `wenmerge.py`. Repo comes with some collected data, roughly 4 weeks in the past with 30 min step. It gets updated when you run the script to create the latest prediction. 

You can set values for starting block and step, delete `result.csv` to collect your own data set. Local node IPC is recommended for this step because the initial data collection can take a while. 

There are two ways to run the prediction script. You can either just print the predicted data or serve it as API. 

For simply printing the prediction and saving created charts to local folder, you can configure it with cli flags:

```
python3 wenmerge.py --ttd 60000000000000000000000 --time 1663596000
```

Other option is to serve predicted data as API for frontend. App will run continuously and update the data. To configure values to predict, put them into he `.env` file. Then run the script via API:

```
uvicorn api:app --reload --host 0.0.0.0 --port 5000
```

The API is served on `localhost:5000/prediction`. Vue.js frontend whhich reads and creates a page with the data can be found in `frontend` directory. 

Here is an example of terminal output: 

```
Updating data at block 14680227 #gets the latest data from the network 
Around 05d18h19m49s left 

Total Terminal Difficulty of 4.805e+17 is expected around  Wed May 4 21:26:52 2022 i.e. between Wed May 4 21:03:08 2022 and Wed May 4 22:38:05 2022
```

If you get it succesfully running and then run into error, just restart and you will get the right result. 

