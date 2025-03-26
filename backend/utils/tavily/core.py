import requests
import os
import re

 
def web_api(query, num_results=5, score_threshold=0.7):
    url = "https://api.tavily.com/search"
 
    payload = {
        "query": query,
        "topic": "news",
        "search_depth": "basic",
        "chunks_per_source": 3,
        "max_results": num_results,
        "time_range": None,
        "days": 3,
        "include_answer": True,
        "include_raw_content": False,
        "include_images": True,
        "include_image_descriptions": True,
        "include_domains": [],
        "exclude_domains": []
    }
 
    headers = {
        "Authorization": f"Bearer {os.getenv('TAVILY_API_KEY')}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    response_data = response.json()
    output={}
    filtered_results = []
    if "results" in response_data:
        high_scoring_results = [
            result for result in response_data["results"]
            if result.get("score", 0) > score_threshold and result.get("content", "").strip().lower() != "failed"
        ]
        high_scoring_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        filtered_results = high_scoring_results[:num_results]
 
        if not filtered_results and response_data["results"]:
            best_result = max(response_data["results"], key=lambda x: x.get("score", 0), default=None)
            if best_result:
                filtered_results = [best_result]
 
        output["filtered_results"] = filtered_results or []
        output["source_ref"] = [
            {"url": urls.get("url", "N/A"), "title": urls.get("title", "N/A")}
            for urls in response_data.get("results", [])
        ]
        output["image_ref"] = response_data.get("images", [])

        output["answer"] = response_data.get("answer", "No answer found.")
    
    return output