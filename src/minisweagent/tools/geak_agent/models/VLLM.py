import os
from typing import List
import openai
from openai import BadRequestError
from tenacity import retry, stop_after_attempt, wait_random_exponential

from geak_agent.models.Base import BaseModel
import json
import re
from loguru import logger

def find_all_thought_indexes(input_str):
    """
    find all index for substring containing "thought"
    
    args:
        input_str (str): the original string
    
    return:
        list: the indexes of all matched strings
    """
    target = '{"thought"' if '{"thought"' in input_str else  '{\n    "thought"'
    target_len = len(target)
    indexes = []
    start = 0  # initial place
    
    while True:
        # searching from the start
        index = input_str.find(target, start)
        if index == -1:  # no match, break
            break
        indexes.append(index)
        # renew the start point
        start = index + target_len
    
    return indexes


def remove_content_between_tk_markers(input_str: str) -> str:
    """
    Remove all content enclosed between the <tk> and </tk> markers (including the markers themselves).
    This function supports handling multiple marker pairs and multi-line content between markers.
    
    Args:
        input_str: The original input string that may contain <tk>...</tk> blocks.
        
    Returns:
        The processed string with all <tk>...</tk> blocks removed.
    """
    # Define the regular expression pattern to match <tk> + any content + </tk>
    # r"<tk>.*?</tk>": Non-greedy match to ensure correct handling of multiple marker pairs
    # re.DOTALL flag: Make the '.' character match newline characters (supports multi-line content between markers)
    tk_pattern = r"<think>.*?</think>"
    processed_string = re.sub(tk_pattern, "", input_str, flags=re.DOTALL)
    
    return processed_string

class VLLMModel(BaseModel):
    """
    OpenAI-compatible client for vLLM servers.

    Configuration precedence for base URL:
      1) base_url argument
      2) env OPENAI_BASE_URL
      3) env VLLM_BASE_URL
      4) env MODEL_API_URL
      5) default http://localhost:8000/v1

    Notes:
    - If your vLLM server requires auth, pass api_key to set Authorization: Bearer <api_key>.
    - If no auth is required, api_key can be omitted; we use a dummy token "EMPTY".
    - messages should be a staandard OpenAI chat list: [{"role": "user|system|assistant", "content": "..."}, ...]
    """

    def __init__(
        self,
        api_url = "http://0.0.0.0:8001/v1/chat/completions",
        max_length = 32768,
        model_name = None,
        api_key = "empty",
    ):
        assert api_url.endswith("chat/completions") or api_url.endswith("v1") or api_url.endswith("v0")
        # Use a dummy key if none provided (vLLM often doesn't require auth)
        if api_url.endswith("chat/completions"):
            resolved_base_url = api_url.split("chat")[0]
        else:
            resolved_base_url = api_url

        # Standard OpenAI client (NOT Azure)
        self.client = openai.OpenAI(
            base_url=resolved_base_url,
            api_key=api_key,
        )
        self.max_length = max_length
        self.model_id = self.client.models.list().data[0].id
        logger.info(f"[VLLMModel] Using api url: {api_url}")
        logger.info(f"[VLLMModel] Using model: {self.model_id}")

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate(
        self,
        messages: List,
        temperature: float = 0,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        max_tokens: int = 32768,
        enable_thinking: bool = False,
        seed = 0,
    ) -> str:


        attempt = 0
        cur_max_tokens = max_tokens

        while attempt < 4:
            # print("###" * 30)
            # print('loop')
            # print("###" * 30)
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=temperature,
                    n=1,
                    stream=False,
                    stop=None,
                    max_tokens=cur_max_tokens,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    logit_bias=None,
                    user=None,
                    seed=seed,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": enable_thinking},
                    }
                )

                # 成功直接返回
                if not response or not hasattr(response, "choices") or len(response.choices) == 0:
                    return None
                response = response.choices[0].message.content
                if '<think>' in response:
                    response = remove_content_between_tk_markers(response)
                thought_indexs = find_all_thought_indexes(response)
                if len(thought_indexs) > 1:
                    response = response[:thought_indexs[1]]
                return response

            except BadRequestError as e:
                msg = str(e)

                if "maximum context length" in msg or "context length" in msg:
                    attempt += 1

                    # 第一次失败：减半再试
                    if attempt < 4:
                        cur_max_tokens = max(1, cur_max_tokens // 2)
                        logger.info(
                            f"[VLLMModel] Context length exceeded. "
                            f"Retrying with max_tokens={cur_max_tokens}"
                        )
                        continue

                    # 第二次仍失败：返回空字符串
                    logger.info(
                        f"[VLLMModel] Context length exceeded after retry. "
                        f"Return empty string."
                    )
                    return ""

                # 不是 context length 的 400，别吞
                logger.info(
                    f"[VLLMModel] BadRequestError. "
                    f"Return empty string. "
                )
                return None

            except Exception:
                return None

        return None