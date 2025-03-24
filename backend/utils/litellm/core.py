from litellm import completion
from datetime import datetime

def llm(model, prompt):
    messages = [{"role": "user", "content": prompt}]
    response = completion(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=1000
    )
    
    return {'id':response.id,
            'prompt': prompt, 
            'markdown': response.choices[0].message.content,
            'model' : response.model,
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens' : response.usage.completion_tokens,
            'created' : datetime.fromtimestamp(response.created).strftime('%Y-%m-%d %H:%M:%S')
            }