from pydantic import BaseModel
import os
from dotenv import load_dotenv
import requests
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
import logging

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

DATABRICKS_TOKEN = os.environ["DATABRICKS_TOKEN"]
DATABRICKS_ENDPOINT_URL = os.environ["DATABRICKS_ENDPOINT_URL"]
API_KEY = os.environ["API_KEY"]
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
def query(request: QueryRequest, key: str = Security(verify_api_key)) -> dict:

    payload = {"dataframe_split": {"columns": ["question"], "data": [[request.question]]}}

    try:
        logger.info("Calling Databricks endpoint")
        response = requests.post(
                DATABRICKS_ENDPOINT_URL, 
                headers={
                    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
                    "Content-Type": "application/json"
                },
                json=payload, 
                timeout=180
            )
        
        response.raise_for_status()
        logger.info("Successful response")
        return {"answer": response.json()["predictions"][0]["answer"]}
        

    except Exception as e:
        logger.exception("Error failed to query Databricks endpoint")
        raise HTTPException(status_code=500, detail = "Failed to reach Databricks endpoint")

@app.get("/health")
def health():
    return {"status": "ok"}
