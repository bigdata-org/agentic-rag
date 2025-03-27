import requests
import os
import json
from langchain.tools import tool
from utils.pytract.core import pytract_rag
from utils.tavily.core import web_api
from utils.snowflake.core import chart_api
from utils.litellm.core import llm
from utils.helper import ra_system_prompt, report_user_prompt
from langchain.tools import tool
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from operator import add
import logging
# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Define AgentState
class State(TypedDict):
    web: dict
    rag: dict
    sf: dict
    available_agents: str
    llm_operations: Annotated[list[dict], add ]
    model_responses: Annotated[list[dict], add]
    web_search_result: str
    rag_search_result: str
    sf_search_result : list
    combined_output: tuple

    
def sf_llm_call(state: State):
    if state['available_agents'] in ['snowflake', 'combined']:
        logger.info('======================================================================') 
        logger.info('snwoflake agent invoked')
        logger.info('======================================================================') 
        input_params = state['llm_operations'][-1]
        response = llm(**input_params)['answer']
        return {'model_responses':[{ 'answer': response}]}
    return {'model_responses':[{ 'answer': json.dumps({'columns':[]})}]}

def agg_llm_call(state: State):
    input_params = state['llm_operations'][-1]
    response = llm(**input_params)['answer']
    return {'model_responses':[{ 'answer': response}]}

def sf_search(state: State):
    input_params = {'raw_json_string':state['model_responses'][0]['answer']}
    chart_data = chart_api(**input_params)
    return {"sf_search_result": chart_data}

def web_search(state: State):
    if state['available_agents'] in ['web','combined']:
        logger.info('======================================================================') 
        logger.info('web agent invoked')
        logger.info('======================================================================') 
        input_params = state['web']
        context = web_api(**input_params)
        return {"web_search_result": context}
    return {"web_search_result": "Web Agent was not invoked"}


def rag_search(state:State) -> str:
    if state['available_agents'] in ['rag','combined']:
        logger.info('======================================================================') 
        logger.info('rag agent invoked')
        logger.info('======================================================================') 
        input_params = state['rag']
        nvidia_rag = pytract_rag()
        context = nvidia_rag.run_nvidia_text_generation_pipeline(**input_params)
        return {"rag_search_result": context}
    return {"rag_search_result": "Rag Agent was not invoked"}

def aggregator(state: State):
    model = state['llm_operations'][0]['model']
    prompt = state['rag']['query']
    rag_result = state['rag_search_result']
    web_result = state['web_search_result']
    logger.info('======================================================================') 
    logger.info('Final LLM Context Window')
    logger.info(report_user_prompt.format(rag_result, web_result, prompt))
    logger.info('======================================================================') 
    return {"llm_operations": [{"model":model,"user_prompt": report_user_prompt.format(rag_result, web_result, prompt), "system_prompt":ra_system_prompt}]}

def final_report(state:State):
    """Combine the results in to generate a research report"""  
    print('calling final report') 
    report_markdown = state['model_responses'][-1]['answer']
    chart_data = state['sf_search_result']
    return {'combined_output': (report_markdown, chart_data)}

def agent_builder():
    parallel_builder = StateGraph(State)
    parallel_builder.add_node("rag_search", rag_search)
    parallel_builder.add_node("web_search", web_search)
    parallel_builder.add_node("sf_search", sf_search)
    parallel_builder.add_node("sf_llm_call", sf_llm_call)
    parallel_builder.add_node("agg_llm_call", agg_llm_call)
    parallel_builder.add_node("aggregator", aggregator)
    parallel_builder.add_node("final_report", final_report)

    parallel_builder.add_edge(START, "rag_search")
    parallel_builder.add_edge(START, "web_search")
    parallel_builder.add_edge(START, "sf_llm_call")  
    parallel_builder.add_edge("sf_llm_call", "sf_search") 

    parallel_builder.add_edge(["rag_search", "web_search"], "aggregator")
    parallel_builder.add_edge("aggregator", "agg_llm_call") 
    parallel_builder.add_edge(["sf_search", "agg_llm_call"], "final_report")  # Combine everything at final report
    parallel_builder.add_edge("final_report", END)
    
    parallel_workflow = parallel_builder.compile()
    return parallel_workflow

def invoke_agent(agent, initial_state):
    output_state = agent.invoke(initial_state)
    return output_state.get('combined_output', None)