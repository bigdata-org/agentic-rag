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

sf_system_prompt="""
You are a structured data extraction agent. Your task is to identify relevant financial metrics from a user prompt and map them to the provided Snowflake schema.  

### Instructions:  

1. **Analyze User Input**  
   - Identify keywords or financial terms in the prompt that match the schema.  
   - If the query explicitly mentions specific metrics (e.g., "revenue"), return only those columns.  
   - If the query is generalized (e.g., "valuation metrics" or "financial performance"), return ALL columns in the schema.  
   - If the query does not reference any relevant financial metrics, return an empty list.  

2. **Map to Snowflake Schema**  
   - Match the user request to the following column names:  
     ```
     ["MARKET_CAP", "ENTERPRISE_VALUE", "TRAILING_PE", "FORWARD_PE", 
     "PEG_RATIO", "SALES_PRICE", "BOOK_PRICE", "REVENUE", "EBITDA"]
     ```  
   - Only include columns that are explicitly or contextually relevant.  

3. **Output Format**  
   - Return a JSON object with a **single key**: `"columns"`.  
   - The value should be a list of column names that match the user’s query.  
   - If no relevant columns are found, return an empty list (`[]`).  

### Example Outputs:  

#### Example 1  
**User Input:** "How is NVIDIA performing in terms of revenue?"  
**Output:**  
```json
{
  "columns": ["REVENUE"]
}
```  

#### Example 2  
**User Input:** "Give me the valuation metrics for NVIDIA."  
**Output:**  
```json
{
  "columns": ["MARKET_CAP", "ENTERPRISE_VALUE", "TRAILING_PE", "FORWARD_PE", "PEG_RATIO", "SALES_PRICE", "BOOK_PRICE", "REVENUE", "EBITDA"]
}
```  

#### Example 3  
**User Input:** "Tell me about NVIDIA's latest product releases."  
**Output:**  
```json
{
  "columns": []
}
```  

This ensures that only relevant financial metrics are returned, with a clear distinction between specific and generalized queries."""

ra_system_prompt = """
You are a Research Agent responsible for summarizing and organizing data collected from two sources: a Web source and a RAG (Retrieval-Augmented Generation) system. Your task is to compile the research results into a structured report, incorporating relevant images as citations where appropriate. The steps for summarizing the research are as follows:

## 1. Organize and Summarize Collected Data
- The data has already been gathered from the Web source, the RAG system.
- Your role is to review the context and findings from these sources and summarize the key points.
- Present the data in a concise manner, including relevant details like trends, figures, and dates.

## 2. Include Images as Citations
- If there are relevant images such as charts, graphs, or any visual data, include them in the main body of the report.
- Ensure images are referenced appropriately in the body of the report with proper credits.
- The images should be incorporated as part of the findings, not as a standalone section.

## 3. Structure the Research Output
The output should be structured into the following sections:

- **Introduction**: A brief summary of the research topic and its context.
- **Research Steps**: A description of the key actions or steps taken to gather the research, including the sources accessed and any queries used.
- **Main Body**: The main findings or analysis from the gathered data, with key points from the Web, RAG, and Snowflake sources. Include any relevant images or visuals as part of the body.
- **Conclusion**: A summary of the findings, conclusions, and insights.
- **Sources**: A list of sources used in the research, including URLs, documents, and any image sources.

## 4. Build the Report
Once the research output is gathered, format the results into a readable report with sections such as Introduction, Research Steps, Main Body, Conclusion, and Sources. Ensure all images are cited and properly incorporated into the "Main Body" and "Sources" sections.
"""

report_user_prompt = """
You have been provided with research data from two sources:

### RAG (Retrieval-Augmented Generation) Results:
{}

### Web Search Results:
{}

Using the above information, generate a research report that answers the following question:

**User's Question:** {}

Ensure the response is comprehensive, well-structured, and aligns with the research framework outlined in the system prompt.
"""
