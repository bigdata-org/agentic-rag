from utils.helper import *
from utils.litellm.core import llm
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
        
        response = llm(model='gemini/gemini-1.5-pro', prompt='hi who built you')
        if 'markdown' in response:
            return {"markdown": response['markdown']}
        return handle_internal_server_error()
    except HTTPException as e:
        raise e
    except Exception as e:
        raise e
