from fastapi import FastAPI
from wenmerge import *
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware  

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if 'TIME_TARGET' in os.environ:
	target=(int(os.environ['TIME_TARGET']))
	prediction=estimate_time(target)

if 'TTD_TARGET' in os.environ:
	target_ttd=(int(os.environ['TTD_TARGET']))
	prediction=estimate_ttd(target_ttd)
    

@app.get("/prediction")
async def predicted_value():
    return time.ctime(prediction)

@app.get("/target")
async def target_value():
    return target_ttd
