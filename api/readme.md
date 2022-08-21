# API for serving predictions

In addition to simply running the script locally, you might need to publish estimates and other crunched data. Here, you have the option to provide data via an API for the front end to use for this purpose.  It's FastAPI which calls on-demand functions from modified predictions script.
This is unfinished effort in MVP stage which has been stalled in favor of a simpler approach.

To serve the API, make sure you have installed the requirements and configure values to predict. In this case, you have to put values into the `.env` file. Then run the script for serving the API: 
```
uvicorn api:app --reload --host 0.0.0.0 --port 5000
```

App is now served locally and you can test it with a simple call: 

```
curl localhost:5000/ttd_prediction
"Thu Sep 15 02:05:36 2022"
```

Details on available API endpoints can be found at `localhost:5000/docs`. 

Check [frontend](../frontend/) directory to run a vuejs frontend that reads data from the API and displays it. 



