# AI-Powered Integrated Research Assistant (Agentic RAG)

## Overview
This Proof of Concept (PoC) demonstrates the feasibility of an AI-powered Integrated Research Assistant designed to generate comprehensive research reports on NVIDIA. The system consolidates structured financial data, historical performance insights, and real-time industry trends using a multi-agent architecture.

## System Architecture
The solution is built using a multi-agent system orchestrated with LangGraph, integrating:

### Agents
- **Snowflake Agent** – Queries and summarizes structured financial data (valuation metrics from Yahoo Finance stored in Snowflake).
- **RAG Agent (Pinecone-powered)** – Retrieves historical insights from NVIDIA's quarterly reports using metadata-filtered search (Year, Quarter).
- **Web Search Agent** – Fetches real-time industry news from APIs like SerpAPI, Tavily, or Bing.

These agents interact to synthesize a well-rounded research report, incorporating textual summaries, visualizations, and real-time insights.

## Implementation Approach
### Data Ingestion & Storage
- **Structured Data:** Extracted from Yahoo Finance and stored in Snowflake.
- **Unstructured Data:** NVIDIA's quarterly reports are chunked and indexed in Pinecone with metadata.
- **Web Data:** Retrieved dynamically using a search API.

### Processing & Analysis
- **Pinecone-based RAG** for historical trend retrieval.
- **Snowflake queries** for financial metrics and visualizations.
- **LLM-based response generation** for summarizing results.

### User Interaction
- **Streamlit Interface** enables user queries with filtering options (Year/Quarter).
- **Users can trigger** individual or combined agent responses for tailored insights.

## Project Links
- **GitHub Repository:** [Agentic RAG](https://github.com/bigdata-org/agentic-rag)
- **Streamlit Frontend:** [Live Demo](https://nvidia-agentic-rag.streamlit.app/)
- **Project Demonstration Video:** Call with Big Data-20250228_050702-Meeting Recording.mp4
- **Backend API Endpoint:** [Agentic RAG Backend](https://agentic-rag-451496260635.us-central1.run.app/heartbeat)

## Getting Started
### Prerequisites
- Python 3.8+
- Snowflake Account
- Pinecone API Key
- Search API Key (SerpAPI, Tavily, or Bing)

### Installation
Clone the repository:
```bash
git clone https://github.com/bigdata-org/agentic-rag.git
cd agentic-rag
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Project
1. Set up environment variables for API keys and database credentials.
2. Start the backend service:
   ```bash
   python app.py
   ```
3. Run the Streamlit frontend:
   ```bash
   streamlit run app.py
   ```

## Contributing
Contributions are welcome! Feel free to submit issues or pull requests to enhance the project.

## License
This project is licensed under the MIT License.
