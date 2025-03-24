from utils.helper import *
from utils.pytract.core import pytract_rag
from utils.litellm.core import llm
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
from  dotenv import load_dotenv
import logging
from typing import List, Dict, Any, Optional

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)
load_dotenv()

app = FastAPI()

class qaModel(BaseModel):
    year: str
    qtr: str
    model: str
    prompt: str
    
@app.post('/qa') 
async def qa_pipeline(request: qaModel):
    try:
        year = request.year
        qtr = request.qtr
        model = request.model
        prompt = request.prompt
        
        logger.info(f"Year: {year}, Quarter: {qtr}, Model: {model},Prompt: {prompt}")
        
        if invalid_model(model):
            raise handle_invalid_model()
        if invalid_prompt(prompt):
            raise handle_invalid_prompt()
        
        # nvidia_rag = pytract_rag()
        # response = nvidia_rag.run_nvidia_text_generation_pipeline([{"year":year, "qtr":qtr}], query=prompt, model=model)
        response = {"markdown":"check your code you've commented the llm part of the application"}
        if "markdown" in response:
            return {"markdown": response['markdown']}
        return handle_internal_server_error()
    except HTTPException as e:
        raise e
    except Exception as e:
        raise e
    
    
@app.get("/heartbeat")
async def heartbeat():
    return {"status": "healthy", "timestamp_ns": time.time_ns()}