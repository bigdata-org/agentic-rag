from pathlib import Path
from fastapi import  HTTPException
    

def invalid_prompt(prompt: str) -> bool:
    if not prompt or not isinstance(prompt, str):
        return True  # Invalid prompt
    if len(prompt.strip()) == 0:  # Empty or whitespace-only prompt
        return True
    if len(prompt) > 1000:  # Example constraint: Max length of 500 characters
        return True
    return False

def invalid_model(model: str) -> bool:
    if not model or not isinstance(model, str):
        return True  # Invalid prompt
    if len(model.strip()) == 0:  # Empty or whitespace-only prompt
        return True
    return False

def handle_internal_server_error(detail="Internal Server Error"):
    raise HTTPException(status_code=500, detail=detail)

def handle_invalid_prompt():
    raise HTTPException(status_code=400, detail="Invalid prompt. Please provide a valid text prompt.")

def handle_invalid_model():
    raise HTTPException(status_code=400, detail="Invalid Model. Please provide a valid model.")