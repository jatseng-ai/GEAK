from typing import List
import requests
from tenacity import retry, stop_after_attempt, wait_random_exponential
from geak_agent.models.Base import BaseModel


class ClaudeModel(BaseModel):
    def __init__(self, 
                 model_id="claude-3.7", 
                 api_key=None):
        assert api_key is not None, "no api key is provided."
        self.model_id = model_id
        self.SERVER = "https://llm-api.amd.com/claude3"
        self.headers = {
            'Ocp-Apim-Subscription-Key': api_key
        }
    
    @retry(wait=wait_random_exponential(min=20, max=60), stop=stop_after_attempt(5))
    def generate(self, 
                 messages: List,
                 temperature=1.0,
                 presence_penalty=0, 
                 frequency_penalty=0, 
                 max_tokens=30000,
                 seed=0
                 ) -> str:
        body = {
            "messages": messages,
            "temperature": temperature,
            "stream": False,
            "max_completion_tokens": max_tokens,
            "max_tokens": max_tokens,
            "presence_Penalty": presence_penalty,
            "frequency_Penalty": frequency_penalty,
            "seed": seed
        }
        try:
            response = requests.post(
                        url=f"{self.SERVER}/{self.model_id}/chat/completions",
                        json=body,
                        headers=self.headers,
                        timeout=600
                    )
        except Exception as e:
            raise ValueError("No response from the API.")
        assert response.status_code == 200
        return response.json()['content'][0]['text']
    