import re
import json
import ast

def extract_function_signatures(code):
    function_defs = []
    pattern = r'def\s+([a-zA-Z0-9_]+)\s*\(([^)]*)\)'
    matches = re.finditer(pattern, code)
    
    for match in matches:
        func_name = match.group(1)
        params = match.group(2)
        function_defs.append(f"def {func_name}({params})")
    
    return function_defs

def clear_code(code):
    if  "```python" in code:
        code = code.split("```python")[-1].replace("<|im_end|>", "").replace("<|EOT|>", "")
    if "```" in code:
        code = code.split("```")[0]
    return code

def extract_function_calls(code):
    calls = []
    pattern = r'([a-zA-Z0-9_]+)\s*\(([^)]*)\)'
    matches = re.finditer(pattern, code)
    
    for match in matches:
        func_name = match.group(1)
        args = match.group(2)
        calls.append(f"{func_name}({args})")
    
    return calls

def clear_json(response):
    if type(response) is dict:
        return response
    elif type(response) is not str:
        response = str(response)
    try:
        response = response.replace("\n", " ")
        response = re.search('({.+})', response).group(0)
        response = re.sub(r"(\w)'(\w|\s)", r"\1\\'\2", response)
        result = ast.literal_eval(response)
    except (SyntaxError, NameError, AttributeError):
        return "ERR_SYNTAX"
    return result

def extract_json_from_stdout_benchmark(stdout_string: str) -> dict:
    """
    Extracts a JSON object from a string that contains json in markdown format e.g. ```json\n .... \n```.

    Args:
        stdout_string: The complete stdout output as a string.

    Returns:
        A dictionary containing the extracted JSON data, or an empty dictionary
        if no valid JSON is found.
    """
    # Use a regular expression to find the content between the ```json
    # and the closing ```.
    pattern = re.compile(r'```json\s*(.*?)\s*```', re.DOTALL)

    match = pattern.search(stdout_string)

    if match:
        json_string = match.group(1).strip()
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            return {}
    else:
        return {}


def extract_json_from_stdout_benchmark(stdout_string: str) -> dict:
    """
    Extracts a JSON object from a string that contains json in markdown format e.g. ```json\n .... \n```.

    Args:
        stdout_string: The complete stdout output as a string.

    Returns:
        A dictionary containing the extracted JSON data, or an empty dictionary
        if no valid JSON is found.
    """
    # Use a regular expression to find the content between the ```json
    # and the closing ```.
    pattern = re.compile(r'```json\s*(.*?)\s*```', re.DOTALL)

    match = pattern.search(stdout_string)

    if match:
        json_string = match.group(1).strip()
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            return {}
    else:
        return {}


def extract_json_from_stdout(input_str, old_latency=[]):
    """
    """
 
    lines = input_str.split('\n')
    
    if "Perf" in input_str and "ms" in input_str:
        pattern = r'Perf(?:ormance)?:\s*(-?\d+\.?\d*)\s*ms'
        reverse_flag = False
    else:
        pattern = r'test cases, avg\s*(-?\d+\.?\d*)\s*G'
        reverse_flag = True

    
   
    numbers = []
    for line in lines:
  
        line = line.strip()
        if not line:  
            continue
        

        match = re.search(pattern, line, re.IGNORECASE)  
        if match:
            try:
                num = float(match.group(1))
                if reverse_flag:
                    num = 1/num
                numbers.append(num)
            except ValueError:

                continue
    output_dict = {"latency": numbers}
    if len(old_latency) > 0:
        assert len(old_latency) == len(numbers), f"Latency error! the number of latency must be the same!, but got {old_latency} and {numbers}"
        speedups = [y/x for x, y in zip(numbers, old_latency)]
        speedup = sum(speedups)/len(speedups)
        output_dict["speedup"] = speedup
    else:
        output_dict["speedup"] = 1.0
        

    return output_dict