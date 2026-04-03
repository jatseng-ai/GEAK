from Levenshtein import distance as levenshtein_dist
import re
import numpy as np
from loguru import logger
import os

def replace_function_body(code: str, func_name: str, input_kernel: str, new_body: str) -> str:
    assert input_kernel in code,"input kernel must in code!"
    new_kernel = input_kernel[:input_kernel.index('{')+1] +'\n    ' + new_body + '\n}'
    new_code = code.replace(input_kernel, new_kernel)
    return new_code

def extract_kernel_body(kernel_code: str) -> str:
    """
    extract the kernel body from the kernel implementation

    
    args:
        kernel_code: str -> full implementation of a kernel (including __global__ void ... { ... })
    
    返回:
        str -> the kernel body (remove the outmost { })
    """
    # find the first '{' corresponding '}'
    start = kernel_code.find('{')
    if start == -1:
        #raise ValueError("No opening brace '{' found in kernel code.")
        #print(kernel_code)
        logger.info("[WARNING] No matching opening brace '{' found. return full code")
        return kernel_code

    end = -1
    brace_count = 0
    doc_string_flag = False
    
    for i in range(start, len(kernel_code)):
        if kernel_code[i] == '/' and kernel_code[i+1] == '/':
            doc_string_flag = True
        
        if doc_string_flag and kernel_code[i] == '\n':
            doc_string_flag = False

        if kernel_code[i] == '{' and not doc_string_flag:
            brace_count += 1
        elif kernel_code[i] == '}' and not doc_string_flag:
            brace_count -= 1
            if brace_count == 0:
                end = i
                break
    if i == len(kernel_code)-1 and end == -1:
        #print(kernel_code)
        logger.info("[WARNING] No matching closing brace '}' found. return full code")
        end = -1
        #raise ValueError("No matching closing brace '}' found.")
    else:
        pass
        #logger.info('successfully extract kernel body')
    # got the content inside the outmost {}
    body = kernel_code[start + 1:end].strip()
    return body


def parse_config_defines_new(config_path: str, json_path="/definition_dict.json") -> dict:
    """
    Parse the config.hpp file and extract #define macros that map 
    ROCPRIM_* keywords to their underlying values like __device__, __shared__, etc.
    
    Returns a dictionary where keys are the macro names and values are their replacements.
    """
    import json
    if os.path.exists(json_path):
        define_dict = json.loads(open(json_path).read())
        return define_dict

    from minisweagent.tools.geak_agent.models.Claude import  ClaudeModel

    model =  ClaudeModel(model_id="claude-opus-4.5", api_key="bdecd15bd6a348d9bb8f7daf6cd023b1")
    code_text = open(config_path).read()
    prompt = f"""
    Task: Convert valid #define macros from C++ code to a Python dictionary.
    Instructions:
    1. Read the C++ code content provided below (contains #define directives).
    2. Filter out empty #define macros (macros with no value after the macro name, e.g., "#define ROCPRIM_CONFIG_HPP_").
    3. For each non-empty #define line (e.g., #define ROCPRIM_DEVICE __device__), extract the macro name as the key and the definition as the value.
    4. Output ONLY the Python dictionary (no other text, comments, or formatting).

    C++ code content:
    {code_text}
    """
    msg = [{"role": "user", "content": prompt}]
    result = model.generate(msg, temperature=1.0, max_tokens=32768, seed=0)
    #json.dumps(open('tmp.json', result))
    with open(json_path,'w') as writer:
        writer.write(result)
    
    define_dict = json.loads(open(json_path).read())
    return define_dict


def parse_config_defines(config_path: str) -> dict:
    """
    Parse the config.hpp file and extract #define macros that map 
    ROCPRIM_* keywords to their underlying values like __device__, __shared__, etc.
    
    Returns a dictionary where keys are the macro names and values are their replacements.
    """
    define_dict = {}
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Pattern to match #define ROCPRIM_* statements with various value types
    # Examples:
    # #define ROCPRIM_DEVICE __device__
    # #define ROCPRIM_HOST __host__
    # #define ROCPRIM_HOST_DEVICE __host__ __device__
    # #define ROCPRIM_SHARED_MEMORY __shared__
    # #define ROCPRIM_FORCE_INLINE __attribute__((always_inline))
    # #define ROCPRIM_INLINE inline
    # #define ROCPRIM_KERNEL __global__ __attribute__((__visibility__("hidden")))
    
    # Match single keyword defines like:
    # #define ROCPRIM_DEVICE __device__
    # #define ROCPRIM_HOST __host__
    # #define ROCPRIM_SHARED_MEMORY __shared__
    single_keyword_pattern = re.compile(
        r'^\s*#define\s+(ROCPRIM_(?:DEVICE|HOST|SHARED_MEMORY))\s+(__\w+__?)\s*$',
        re.MULTILINE
    )
    
    for match in single_keyword_pattern.finditer(content):
        macro_name = match.group(1)
        macro_value = match.group(2).strip()
        define_dict[macro_name] = macro_value
    
    # Match HOST_DEVICE which has two keywords
    # #define ROCPRIM_HOST_DEVICE __host__ __device__
    host_device_pattern = re.compile(
        r'^\s*#define\s+(ROCPRIM_HOST_DEVICE)\s+(__host__\s+__device__)\s*$',
        re.MULTILINE
    )
    
    for match in host_device_pattern.finditer(content):
        macro_name = match.group(1)
        macro_value = match.group(2).strip()
        define_dict[macro_name] = macro_value
    
    # Handle ROCPRIM_INLINE -> inline
    inline_pattern = re.compile(
        r'^\s*#define\s+(ROCPRIM_INLINE)\s+(inline)\s*$',
        re.MULTILINE
    )
    
    for match in inline_pattern.finditer(content):
        macro_name = match.group(1)
        macro_value = match.group(2).strip()
        define_dict[macro_name] = macro_value
    
    # Handle ROCPRIM_FORCE_INLINE -> __attribute__((always_inline))
    force_inline_pattern = re.compile(
        r'^\s*#define\s+(ROCPRIM_FORCE_INLINE)\s+(__attribute__\s*\(\(always_inline\)\))\s*$',
        re.MULTILINE
    )
    
    for match in force_inline_pattern.finditer(content):
        macro_name = match.group(1)
        macro_value = match.group(2).strip()
        define_dict[macro_name] = macro_value
    
    # Handle ROCPRIM_KERNEL -> __global__ __attribute__((__visibility__("hidden")))
    kernel_pattern = re.compile(
        r'^\s*#define\s+(ROCPRIM_KERNEL)\s+(__global__\s+__attribute__\s*\(\(__visibility__\("hidden"\)\)\))\s*$',
        re.MULTILINE
    )
    
    for match in kernel_pattern.finditer(content):
        macro_name = match.group(1)
        macro_value = match.group(2).strip()
        define_dict[macro_name] = macro_value
    
    return define_dict

def replace_defines_forward(code: str, define_dict: dict) -> str:
    """
    Replace the defined keywords (e.g., ROCPRIM_DEVICE) with their original values 
    (e.g., __device__) to allow kernel extraction.
    
    Args:
        code: The source code containing the defined keywords
        define_dict: Dictionary mapping macro names to their values
        
    Returns:
        Code with all macros replaced by their underlying values
    """
    replaced_code = code
    if define_dict is None:
        return replaced_code
    
    # Sort keys by length (longest first) to avoid partial replacements
    sorted_keys = sorted(define_dict.keys(), key=len, reverse=True)
    
    for key in sorted_keys:
        # Use word boundary to avoid partial matches
        pattern = r'\b' + re.escape(key) + r'\b'
        replaced_code = re.sub(pattern, define_dict[key], replaced_code)
    
    return replaced_code

def replace_defines_backward(code: str, define_dict: dict) -> str:
    """
    Replace the original keywords (e.g., __device__) back to defined keywords 
    (e.g., ROCPRIM_DEVICE) after kernel extraction.
    
    Args:
        code: The source code with original keywords
        define_dict: Dictionary mapping macro names to their values
        
    Returns:
        Code with underlying values replaced back to macro names
    """
    replaced_code = code
    if define_dict is None:
        return replaced_code
    
    # Create a reverse dictionary: value -> key
    # Sort by value length (longest first) to handle cases like __host__ __device__ before __device__
    reverse_dict = {}
    for key, value in define_dict.items():
        reverse_dict[value] = key
    
    sorted_values = sorted(reverse_dict.keys(), key=len, reverse=True)
    
    for value in sorted_values:
        # Use word boundary or pattern matching for proper replacement
        # For attributes like __attribute__((always_inline)), need exact match
        pattern = re.escape(value)
        replaced_code = re.sub(pattern, reverse_dict[value], replaced_code)
    
    return replaced_code

def extract_hip_kernels_core(code: str):
    """
    extract the kernel code based on the brackets '{' and '}'
    Supports both __global__ void and ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void patterns
    """
    # Extended pattern to match both standard HIP kernels and ROCPRIM-style device functions
    # Pattern 1: __global__ void kernel_name(...)
    # Pattern 2: __device__ __attribute__((always_inline)) void func_name(...)
    # Pattern 3: ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void func_name(...)
    kernels = []
    template_pattern = re.compile(
        r'(template\s*<[^>]*>\s*\n?\s*)'  # template<...>
        r'((?:__global__|__device__|__host__\s+__device__|inline)\s*)+\s*'  # device qualifiers (after replacement)
        r'(\w+(?:\s*::\s*\w+)*(?:\s*<[^>]*>)?)\s+'  # return type (can be templated)
        r'([A-Za-z_]\w*)\s*'  # function name
        r'\([^{]*\)\s*'  # parameters (can span multiple lines)
        r'\{',  # opening brace
        re.DOTALL
    )
    
    for match in template_pattern.finditer(code):
        name = match.group(4)
        start = match.start()
        brace_start = match.end() - 1  # index for '{'

        # matching for '{' and '}'
        brace_count = 0
        i = brace_start
        while i < len(code):
            if code[i] == '{':
                brace_count += 1
            elif code[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break
            i += 1
        else:
            end = len(code)

        full_func = code[start:end]
        signature = code[match.start():brace_start+1].strip()
        kernels.append({
            "name": name,
            "signature": signature,
            "body": full_func.strip()
        })
    header_pattern = re.compile(
        r'(?:__global__|__device__|ROCPRIM_DEVICE)\s+'
        r'(?:__launch_bounds__\s*\([^)]*\)\s+)?'
        r'(?:__attribute__\s*\(\([^)]*\)\)\s+|ROCPRIM_FORCE_INLINE\s+)?'
        r'(?:inline\s+)?'
        r'void\s+'
        r'([A-Za-z_]\w*)'
        r'\s*\([^)]*\)\s*\{',
        re.DOTALL
    )

    for match in header_pattern.finditer(code):
        name = match.group(1)
        start = match.start()
        brace_start = match.end() - 1  # index for '{' 
        # Check if this function is already captured by template pattern
        already_captured = False
        for k in kernels:
            if k["name"] == name and start >= code.find(k["body"][:50]):
                already_captured = True
                break
        
        if already_captured:
            continue
        # matching for '{' and '}'
        brace_count = 0
        i = brace_start
        while i < len(code):
            if code[i] == '{':
                brace_count += 1
            elif code[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break
            i += 1
        else:
            end = len(code)  # no matching 

        full_func = code[start:end]
        signature = code[match.start():brace_start+1].strip()
        kernels.append({
            "name": name,
            "signature": signature,
            "body": full_func.strip()
        })
    
    return kernels

def extract_hip_kernels(code: str, kernel_func_name, init=True):
    hip_kernels = extract_hip_kernels_core(code) 
    sel_ind = 0
    if len(hip_kernels) > 0: 
        # original, get the maximun length of the kernel func
        sel_ind = 0
        cur_max = len(hip_kernels[0]['body'])

        for idx in range(1, len(hip_kernels)):
            #cody adds another constrainet that it should contain the kernel func name
            if len(hip_kernels[idx]['body']) > cur_max and kernel_func_name in hip_kernels[idx]['body']:
                sel_ind = idx
                cur_max = len(hip_kernels[idx]['body'])
        # Also check if kernel_func_name matches directly
        for idx in range(len(hip_kernels)):
            if hip_kernels[idx]['name'] == kernel_func_name:
                sel_ind = idx
                break
    else:
        if init:
            raise ValueError('[Warning] cannot extract a hip kernel from the given test case, please check!')
        else:
            logger.info('[Warning] cannot extract a hip kernel from the given test case, please check!')
            return None, None
    input_kernel = hip_kernels[sel_ind]['body']
    kernel_func_name = hip_kernels[sel_ind]['name']
    return input_kernel, kernel_func_name

def normalized_levenshtein(s1, s2):
    """the edit distance for a single string, normalized to [0,1]"""
    if len(s1) == 0 and len(s2) == 0:
        return 0.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0
    
    return levenshtein_dist(s1,s2)/ max_len

def dtw_string_distance(list1, list2):
    """the dtw distance based on edit distance, normalized to [0,1]"""
    # firstly, filter the blank lines in these lists
    list1 = [s.strip() for s in list1 if s.strip()]
    list2 = [s.strip() for s in list2 if s.strip()]
    # also remove the line with initilized with '//', full docstring
    list1 = [s for s in list1 if s.strip()[:2] != '//']
    list2 = [s for s in list2 if s.strip()[:2] != '//']
    #remove all the blank space in each line
    list1 = [s.replace(' ','') for s in list1]
    list2 = [s.replace(' ','') for s in list2]

    m, n = len(list1), len(list2)
    if m == 0 and n == 0:
        return 0.0
    # construct the distance matrix
    dist_matrix = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            dist_matrix[i][j] = normalized_levenshtein(list1[i], list2[j])
    
    # dynamic programming for dtw distance
    dtw = np.full((m+1, n+1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = dist_matrix[i-1, j-1]
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    
    # normalization
    max_len = max(m, n)
    return dtw[m, n] / max_len if max_len != 0 else 0.0