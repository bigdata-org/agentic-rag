import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

# Backend FastAPI URL (Ensure this is accessible from Streamlit Cloud)
BACKEND_URL = "https://agentic-rag-451496260635.us-central1.run.app"  # Adjust as needed
# BACKEND_URL = "http://localhost:8000"

model_mapper = {
    "openai/gpt-4o": "gpt-4o-2024-08-06",
    "openai/gpt-3.5-turbo": "gpt-3.5-turbo-0125",
    "openai/gpt-4": "gpt-4-0613",
    "openai/gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "gemini/gemini-1.5-pro": "gemini/gemini-1.5-pro",
    "gemini/gemini-2.0-flash-lite": "gemini/gemini-2.0-flash-lite"
}

available_models = list(model_mapper.keys())
available_agents  = ['web', 'rag', 'snowflake', 'combined']
# Function to call the query API
def rag(year, qtr, model, prompt, rag_top_k, web_top_k, web_threshold, selected_agent):
    url = f"{BACKEND_URL}/qa"
    data = {
        "year": year,
        "qtr": qtr.replace("Q", ""),  # Strip "Q" from quarter
        "model": model_mapper[model],
        "prompt": prompt,
        "rag_top_k": rag_top_k,
        "web_top_k": web_top_k,
        "web_threshold": web_threshold,
        "available_agents": selected_agent
    }
    
    try:
        response = requests.post(url, json=data, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.ConnectionError:
        st.error("Error: Unable to connect to the backend. Please ensure the server is running and accessible.")
    return None  

# Streamlit UI
st.title("NVIDIA Agentic RAG Explorer")

mode = 'nvidia'
st.header("Financial Report Query")

# Create side-by-side columns for Year, Quarter, and Model selection
col1, col2, col3 = st.columns([1, 1, 1.5])

with col1:
    year = st.selectbox("Select Year:", [str(yr) for yr in range(2021, 2026)], index=4)

with col2:
    qtr = st.selectbox("Select Quarter:", ['Q1', 'Q2', 'Q3', 'Q4'], index=0)

with col3:
    model_choice = st.selectbox("Select available models:", available_models, index=available_models.index("gemini/gemini-1.5-pro"))

with st.sidebar:
    st.markdown("### Agent Configurations")  # Sidebar heading
    st.write("")  # Adds one newline for spacing
    rag_top_k = st.slider("RAG Top K Results", 1, 10, 5)
    web_top_k = st.slider("WEB API Top K Results", 1, 5, 5)
    web_threshold = st.slider("WEB API Score Threshold", 0.0, 1.0, 0.3, step=0.01)
    selected_agent = st.selectbox("Select Agent:", available_agents, index=available_agents.index("combined"))

# User input for query
query = st.text_area("Enter your query:", "", height=150)

# Button to submit the query
if st.button("Submit Query"):
    if query.strip():
        with st.spinner("Querying the backend..."):
            result = rag(year, qtr, model_choice, query, rag_top_k, web_top_k, web_threshold, selected_agent) 
            if result:
                st.session_state['chart_data'] = result['charts']
                st.session_state['markdown'] = result['markdown']
                st.session_state['year'] = year
                st.session_state['qtr'] = qtr.replace("Q", "")
        if result and "markdown" in result:
            pass
        elif result is None:
            st.error("Failed to retrieve data. Please check the backend connection.")
        else:
            st.error("Something went wrong.")
    else:
        st.error("Please enter a valid query.")

if 'markdown' in st.session_state and st.session_state['markdown']:
    st.subheader("Generated Response:")
    st.write(st.session_state["markdown"])

if 'chart_data' in st.session_state and st.session_state['chart_data']:
    st.subheader("Snowflake powered charts:")
    data = st.session_state['chart_data']
    
    for idx, chart_data in enumerate(data):
        df = pd.DataFrame(chart_data)

        # Extract the single metric dynamically (excluding 'year' and 'qtr')
        metric = [col for col in df.columns if col not in ['year', 'qtr']][0]  # Only one metric expected

        # Create a new column for x-axis labels
        df['Year-Qtr'] = df.apply(lambda row: f"{int(row['year'])} - Q{int(row['qtr'])}", axis=1)

        highlight_x = f"{st.session_state['year']} - Q{st.session_state['qtr']}"  # The quarter to highlight
        st.write(highlight_x)
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))
        # Bar colors (highlight specific quarter)
        colors = ['#3498db' if x != highlight_x else '#e74c3c' for x in df['Year-Qtr']]
        # Plot bars
        ax.bar(df['Year-Qtr'], df[metric], color=colors, edgecolor='black', alpha=0.7, label='Bar')
        # Overlay line plot
        ax.plot(df['Year-Qtr'], df[metric], marker='o', linestyle='-', linewidth=2, markersize=6, color='black', label='Trend Line')
        # Labels and title
        ax.set_xlabel("Year - Quarter", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f"{metric} over Time", fontsize=14, fontweight='bold')
        # Rotate x-axis labels for clarity
        plt.xticks(rotation=45)
        # Add legend
        ax.legend()
        # Display in Streamlit
        st.pyplot(fig)
