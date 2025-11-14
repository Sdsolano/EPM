from pydantic import BaseModel
from read import clear_info_power, clear_info_weather
from fastapi import FastAPI, UploadFile, File
import pandas as pd
from io import StringIO
app = FastAPI()

# Modelo Pydantic
class DataRequest(BaseModel):
    data: list

@app.post("/limpiar_power/")
def limpiar_info(request: DataRequest):
    df = pd.DataFrame(request.data)

    df_limpio = clear_info_power(df)

    return "exito"

@app.post("/limpiar_weather/")
def limpiar_info_weather(request: DataRequest):
    df = pd.DataFrame(request.data)

    df_limpio = clear_info_weather(df)

    return "exito"


@app.post("/limpiar_power_csv/")
async def limpiar_power_csv(file: UploadFile = File(...)):
    contents = await file.read()
    s = str(contents, 'utf-8')
    
    df = pd.read_csv(StringIO(s))
    
    df_limpio = clear_info_power(df)
    
    return "exito"

@app.post("/limpiar_weather_csv/")
async def limpiar_weather_csv(file: UploadFile = File(...)):
    contents = await file.read()
    s = str(contents, 'utf-8')
    
    df = pd.read_csv(StringIO(s))
    
    df_limpio = clear_info_weather(df)
    
    return "exito"