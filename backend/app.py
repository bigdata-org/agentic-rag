from utils.helper import *
from utils.pytract.core import pytract_rag
from utils.langgraph.agent import invoke_agent, agent_builder
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
    rag_top_k: int
    web_top_k: int
    web_threshold: float
    
@app.post('/qa') 
async def qa_pipeline(request: qaModel):
    try:
        year = request.year
        qtr = request.qtr
        model = request.model
        prompt = request.prompt
        rag_top_k = request.rag_top_k
        web_top_k = request.web_top_k
        web_threshold = request.web_threshold
        
        logger.info(f"Year: {year}, Quarter: {qtr}, Model: {model},Prompt: {prompt}, rag_top_k : {rag_top_k}, web_top_k, {web_top_k}, web_threshold: {web_threshold}")
        
        if invalid_model(model):
            raise handle_invalid_model()
        if invalid_prompt(prompt):
            raise handle_invalid_prompt()
        
        initial_state = {
        "llm_operations":[{"model": model, "user_prompt": prompt, "system_prompt":sf_system_prompt, "is_json": True}],
        "sf": {"query": prompt},
        "web" :{"query": prompt, "num_results": web_top_k, "score_threshold": web_threshold},
        "rag" : {"search_params": [{"year": year, "qtr": qtr}], "query": prompt, "top_k": rag_top_k}
    }
        agent = agent_builder()
        agent_response = invoke_agent(agent, initial_state)
        
        # response = {"markdown":"check your code you've commented the llm part of the application"}
        response = {"markdown": agent_response[0], "charts": agent_response[1]}
        return response
    except HTTPException as e:
        raise e
    except Exception as e:
        raise e
    
    
@app.get("/heartbeat")
async def heartbeat():
    return {"status": "healthy", "timestamp_ns": time.time_ns()}