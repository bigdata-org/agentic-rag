import streamlit as st
import requests

# Backend FastAPI URL
BACKEND_URL = "http://localhost:8000"  # Adjust as needed

model_mapper = {
    "openai/gpt-4o": "gpt-4o-2024-08-06",
    "openai/gpt-3.5-turbo": "gpt-3.5-turbo-0125",
    "openai/gpt-4": "gpt-4-0613",
    "openai/gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "gemini/gemini-1.5-pro": "gemini/gemini-1.5-pro",
    "gemini/gemini-2.0-flash-lite": "gemini/gemini-2.0-flash-lite"
}

available_models = list(model_mapper.keys())

# Function to call the query API
def rag(year, qtr, model, prompt):
    url = f"{BACKEND_URL}/qa"
    data = {
        "year": year,
        "qtr": qtr.replace("Q", ""),  # Strip "Q" from quarter
        "model": model_mapper[model],
        "prompt": prompt
    }
    response = requests.post(url, json=data)
    return response.json()

# Streamlit UI
st.title("NVIDIA Agentic RAG Explorer")

mode = 'nvidia'
st.header("Financial Report Query")

# Create side-by-side columns for Year and Quarter selection
col1, col2, col3 = st.columns([1, 1, 1.5])  # Adjust column width ratio as needed

with col1:
    year = st.selectbox("Select Year:", [str(yr) for yr in range(2021, 2026)], index=4)  # Default to 2025

with col2:
    qtr = st.selectbox("Select Quarter:", ['Q1', 'Q2', 'Q3', 'Q4'], index=0)  # Default to Q1

with col3:
    model_choice = st.selectbox("Select the data source:", available_models, index=available_models.index("gemini/gemini-1.5-pro"))  # Default to gemini-1.5-pro

# User input for query
query = st.text_area("Enter your query:", "", height=150)

# Button to submit the query
if st.button("Submit Query"):
    if query.strip():  # Ensure the prompt is not empty
        with st.spinner("Querying the backend..."):
            result = rag(year, qtr, model_choice, query)

        if "markdown" in result:
            st.subheader("Generated Response:")
            st.write(result["markdown"])
        else:
            st.error("Something went wrong")
    else:
        st.error("Please enter a valid query.")
