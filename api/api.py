from fastapi import FastAPI
from wenmerge_api import *
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware  

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:80"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if 'TIME_TARGET' in os.environ:
	target=(int(os.environ['TIME_TARGET']))

if 'TTD_TARGET' in os.environ:
	target_ttd=(int(os.environ['TTD_TARGET']))
    

@app.get("/ttd_prediction_timestamp")
async def projected_timestamp_when_ttd_target_is_reached():
    return estimate_ttd(target_ttd)

@app.get("/ttd_prediction")
async def projected_timestamp_when_ttd_target_is_reached():
    return time.ctime(estimate_ttd(target_ttd))

@app.get("/time_prediction")
async def projected_td_at_time_target():
    return estimate_time(target)

@app.get("/target_time")
async def time_target():
    return time.ctime(target)

@app.get("/target_ttd")
async def ttd_target():
    return target_ttd

@app.get("/hashrate")
async def daily_hashrate_and_date_projected_linearly_with_it():
    return estimate_hashrate(target_ttd, target)
