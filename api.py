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


if 'TTD_TARGET' in os.environ:
	prediction=time.ctime(construct_polynom())
	target=(int(os.environ['TTD_TARGET']))
if 'TIME_TARGET' in os.environ:
	prediction=time.ctime(construct_polynom())
	target=(int(os.environ['TIME_TARGET']))



@app.get("/prediction")
async def predicted_value():
    return prediction


@app.get("/target")
async def target_value():
    return target
