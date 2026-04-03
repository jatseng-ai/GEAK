from tqdm import tqdm
import os
import json
import yaml
from typing import List
import subprocess
from loguru import logger
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from geak_agent.agents.reflexion_oneshot import Reflexion_Oneshot
from geak_agent.utils.utils import clear_code, clear_json, extract_json_from_stdout
from geak_agent.utils.utils_ourllm import extract_kernel_body,extract_hip_kernels,replace_function_body, dtw_string_distance, replace_defines_forward, replace_defines_backward
from geak_agent.memories.Memory import MemoryClassMeta
from geak_agent.prompts import prompt_for_generation, prompt_for_reflection
from geak_agent.dataloaders.ProblemState import tempCode, ProblemState
from geak_agent.models.OpenAI import OpenAIModel
from geak_agent.models.Claude import ClaudeModel
from geak_agent.models.Gemini import GeminiModel
from geak_agent.prompts import prompt_for_reflection_hip
from minisweagent.tools.geak_agent.models.VLLM import VLLMModel
from pathlib import Path
import random
import time

class GaAgent_kernel2kernel(Reflexion_Oneshot):
    def __init__(self, config_path):
        #try:
        self.config = self.load_config(config_path)
        if 'gpt' in self.config.model_id:
            model = OpenAIModel(model_id=self.config.model_id, api_key=self.config.api_key)
        elif 'gemini' in self.config.model_id:
            model = GeminiModel(model_id=self.config.model_id, api_key=self.config.api_key)
        elif 'claude' in self.config.model_id:
            model = ClaudeModel(model_id=self.config.model_id, api_key=self.config.api_key)
        else:
            logger.info(f"{self.config.model_id} model is not supported. You can add your model according to script in models or use model existed.")
            sys.exit(1)
        super().__init__(model, self.config.mem_file, self.config.descendant_num)

        #self.generator = ClaudeModel(model_id="claude-opus-4.6", api_key=self.config.api_key)

        self.dtw_threshold = 0.05
        self.max_trial_num = 3
        self.use_ori_code = self.config.use_ori_code
        self.forbid_reflection = self.config.forbid_reflection

    def memory_init(self, mem_file=None, descendant_num=1):
        """
        Args:
            mem_file: previous stored memories, which can be loaded to continue run
        """
        class Memory(metaclass=MemoryClassMeta, field_names=["ps", 
                                                             "call_err_msg", 
                                                             "exe_err_msg",
                                                             "reflection", 
                                                             "perf_candidates",
                                                             "perf_strategy",
                                                             "raw_codes",
                                                             "call_candidate",
                                                             "exe_candidate",
                                                             "temp_strategy",
                                                             "perf_debug_num",
                                                             "pass_call", 
                                                             "pass_exe",
                                                             "pass_perf",
                                                             "history",
                                                             ]):
            pass
        
        if mem_file is not None:
            assert mem_file.endswith(".json"), f"expect a json file, but got {mem_file} instead"
            with open(mem_file, "r") as f:
                input_mems = json.load(f)


        raw_codes =None
        ps = ProblemState
        ps.instruction = "Please optimize the following HIP kernel/function for better performance on the ROCm platform (MI250 GPU).\n    MI250 specs: 208KB LDS per Compute Unit (CU), 64 CUs total.\n\nYou will receive only a single kernel/function from the .hip file.\n    You may only modify the function body, but you must output the entire function including its signature.\n\nAllowed:\n\nRewrite or optimize the function body only.\n\n    Add local variables, shared memory, unrolling, vectorized I/O, etc.\n\nReorder code inside the function.\n\nAdd comments inside the function.\n\nNot Allowed:\n\nDo NOT change the function name.\n\n    Do NOT change the function signature or parameter types.\n\nDo NOT add, remove, or modify any code outside this function.\n\nNo helper functions\n\nNo new includes\n\nNo new kernels\n\n    No changes to launch configuration\n\nDo NOT assume access to any code outside this function.\n\nOptimization guidelines (apply those that fit):\n\nChunked/tiled processing using registers or LDS\n\n    Shared-memory buffering (LDS)\n\nDelayed stores to shared memory\n\nVectorized loads/stores (float2/float4/uint4/etc.)\n\nLoop unrolling\n\nBound checks for variable sizes\n\nMinimize warp/wavefront divergence\n\n    Increase ILP via interleaving independent ops\n\nReduce LDS/register usage for higher occupancy\n\nFavor coalesced memory and AMD wavefront-friendly access patterns\n\nFuse operations where possible\n\n    Use compiler hints like #pragma unroll\n\nHard Requirements:\n\nReturn the full function, including the exact original function signature.\n\nOnly modify code inside the function body.\n\n    Preserve algorithmic correctness and bitwise-equivalent outputs.\n\nMaintains existing formatting and comments unless improving them.\n\nCode must be compilable and runnable."
        if mem_file is None:
            tmp_mem = Memory(ps=ps, 
                            call_err_msg=None,
                            exe_err_msg=None, 
                            reflection=None, 
                            perf_candidates=[],
                            perf_strategy=None,
                            raw_codes=raw_codes,
                            call_candidate=None,
                            exe_candidate=None,
                            temp_strategy=None,
                            perf_debug_num=0,
                            pass_call=False,
                            pass_exe=False,
                            pass_perf=False,
                            history=[[] for _ in range(descendant_num)],
                            )
        else:
            input_mem = input_mems[ps.file_path]
            tmp_mem = Memory(
                ps=ps,
                call_err_msg=input_mem["call_err_msg"],
                exe_err_msg=input_mem["exe_err_msg"], 
                reflection=input_mem["reflection"], 
                perf_candidates=input_mem["perf_candidates"],
                perf_strategy=input_mem["perf_strategy"],
                raw_codes=raw_codes,
                call_candidate=input_mem["call_candidate"],
                exe_candidate=input_mem["exe_candidate"],
                temp_strategy=input_mem["temp_strategy"],
                perf_debug_num=input_mem["perf_debug_num"],
                pass_call=input_mem["pass_call"],
                pass_exe=input_mem["pass_exe"],
                pass_perf=input_mem["pass_perf"],
                history=[[] for _ in range(descendant_num)],
                kernel_code = input_mem['kernel_code'],
                kernel_func_name = input_mem['kernel_func_name']
            )

        self.memories = tmp_mem
    
    def write_memories(self, file_path):
        output_dict = {}
        with open(file_path, "w") as f:
            output = {
                "call_err_msg": str(self.memories.call_err_msg),
                "exe_err_msg": str(self.memories.exe_err_msg),
                "reflection": self.memories.reflection, 
                "perf_candidates": [list(cand) for cand in self.memories.perf_candidates],
                "perf_strategy": self.memories.perf_strategy,
                "call_candidate": self.memories.call_candidate,
                "exe_candidate": self.memories.exe_candidate,
                "temp_strategy": self.memories.temp_strategy,
                "perf_debug_num": self.memories.perf_debug_num,
                "pass_call": self.memories.pass_call, 
                "pass_exe": self.memories.pass_exe,
                "pass_perf": self.memories.pass_perf,
            }
            output_dict[self.memories.ps.file_path] = output
            json.dump(output_dict, f)
    
    def related_file(self):
        related_prompt =""
        if self.config.related_file:
            logger.info("\nrelated file")
            related_prompt ="\nThe following codes are provided as contextual reference only. " \
            "They illustrate how the target code is integrated, instantiated, and executed within the system, " \
            "and should be used to understand constraints and interactions, not as primary optimization targets."
            for file in self.config.related_file:
                if os.path.exists(file):
                    with open(file, "r", encoding="utf-8") as f:
                        file_code = f.read()
                        related_prompt += "\n" + file
                        related_prompt += "\n" + file_code
        return related_prompt

    def run(self, gpu_id=None, code_path=None, test_case=None, kernel_func_name=None, define_dict=None, **kwargs):
        """
        Args:
            
        """
        assert self.config.ancestor_num >= 0, f"expect ancestor_num to be larger than 0, but got {self.config.ancestor_num}"
        assert self.config.descendant_num >= 0, f"expect descendant_num to be larger than 0, but got {self.config.descendant_num}"
        assert self.config.descendant_debug >= 0, f"expect descendant_debug to be larger than 0, but got {self.config.descendant_debug}"
        assert test_case is not None, f"expect test case, but got None"
        if self.config.reset_repo:
            subprocess.run([f"git config --global --add safe.directory {self.config.repo_dir}"], shell=True, cwd=self.config.repo_dir, capture_output=True, text=True, timeout=20)
            subprocess.run(["git restore ."], shell=True, cwd=self.config.repo_dir, capture_output=True, text=True, timeout=20)
        if code_path and os.path.exists(code_path):
            with open(code_path, "r", encoding="utf-8") as f:
                # cody add new args in self.memories.ps
                self.memories.ps.define_dict = define_dict
                full_code = f.read()
                input_kernel, _ = extract_hip_kernels(replace_defines_forward(full_code, self.memories.ps.define_dict), kernel_func_name)
                self.memories.ps.kernel_func_name = kernel_func_name
                self.memories.ps.kernel_code = input_kernel
                self.memories.ps.label = full_code
                self.memories.ps.ori_full_code = full_code
        self.memories.ps.file_path = code_path
        self.memories.ps.test_code = test_case
        self.gpu_id = gpu_id
        self.repo_path = kwargs["repo_path"]
        if gpu_id is not None:
            os.environ["HIP_VISIBLE_DEVICES"] = str(gpu_id)

            if kwargs["gpu_ids"] and kwargs["gpu_ids"].index(gpu_id) == 0:
                logger.info(f"GPU-ID {gpu_id}: directly start")
            else:
                random.seed(int(gpu_id))
                rand_time = random.randint(0,5)
                logger.info(f"GPU-ID {gpu_id}: sleep for {rand_time}")
                time.sleep(rand_time)
        
        # define the generator based on the id
        self.generator = VLLMModel()

    
        #init test, got the baseline performance first
        logger.info(f"Loading original perf for comparison for {kernel_func_name}...")

        cur_path = code_path
        for idx in range(100):
            cur_path = Path(cur_path).parent
            if os.path.exists(os.path.join(str(cur_path), 'baseline_perf.yaml')):
                break
        if idx == 99:
            #raise FileNotFoundError("Cannot find baseline_perf under given code path or its parent! Please check!")
            logger.info("Cannot find baseline_perf under given code path or its parent! rerun the baseline.")
            command = [f"{self.memories.ps.test_code}"]
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=3600)

            output = result.stdout.strip()
            logger.info(output)
            json_output= extract_json_from_stdout(output)
            self.ori_speedup = json_output.get("speedup", 1.0)
            self.ori_latency = json_output.get("latency", [9999])
        else:
            tmp = yaml.safe_load(open(os.path.join(str(cur_path), 'baseline_perf.yaml')))
            self.ori_latency = []
            # for test_case in tmp['test_cases']:
            #     self.ori_latency.append(float(test_case['execution_time_ms']))
            self.ori_latency = tmp['test_cases']
            self.ori_speedup = 1.0

        logger.info(f"GPU-ID {gpu_id}: Original latency set successfully! It is {self.ori_latency}")
        logger.info(f"GPU-ID {gpu_id}: Original speedup set successfully! It is {self.ori_speedup}")

            

        logger.info(f"\n===GPU-ID {gpu_id}: OURLLM Optimize {code_path} ===")
        for iter in range(self.config.iteration_num):
            logger.info(f"\n===GPU-ID {gpu_id}: OURLLM Iteration {iter} ===")
            if self.config.output_path is not None:
                os.makedirs(os.path.dirname(self.config.output_path), exist_ok=True)
                root, extension = os.path.splitext(self.config.output_path)
                mem_output_path = f"{root}_mem_{iter}.json"

            # generate solution
            logger.info(f"\nGPU-ID {gpu_id}:  generate solution")
            self.generate_solution(self.memories, iter)
            
            
            # run scripts
            logger.info(f"\nGPU-ID {gpu_id}:  run scripts on gpu")
            if self.memories.raw_codes:
                for i in range(len(self.memories.raw_codes)):
                    raw_code = self.memories.raw_codes[i]
                    speedup = 0.0
                    latency = 9999
                    if raw_code.pass_perf:
                        continue
                    try:
                        if raw_code.code:
                            command = [f"{self.memories.ps.test_code}"]
                            with open(code_path, 'w') as writer:
                                writer.write(raw_code.code)
                            if self.use_ori_code:
                                speedup = self.ori_speedup
                                latency = self.ori_latency
                                stdout, stderr = "",""
                                pass_call, pass_exe = True, True
                            else:
                                result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=3600)
                                
                                if result.returncode == 0:
                                    output = result.stdout.strip()
                                    pass_call = "Failed to run correctness" not in output and "Failed to run make" not in output
                                    pass_exe = pass_call and "Failed to run performance" not in output
                                    
                                    if pass_exe:
                                        json_output = extract_json_from_stdout(output, self.ori_latency)
                                        speedup = json_output.get("speedup", 0)  
                                        latency = json_output.get("latency", [9999])  
                                        #logger.info(f"GPU-ID {gpu_id}: json output {json_output}")                                
                                    stdout, stderr = result.stdout, result.stderr
                                else:
                                    pass_call, pass_exe, speedup, latency, stdout, stderr = False, False, 0.0, [9999],  result.stdout, result.stderr
                        else :
                            pass_call, pass_exe, speedup, latency, stdout, stderr = False, False, 0.0, 9999, "", "Code is empty"
                            
                        if type(latency) is list and len(latency) < 5:
                            logger.info(f'GPU-ID {gpu_id}: iter {iter}, descendant {i}: pass_call {pass_call}, pass_exe {pass_exe}, latency {latency}, speedup {speedup}')
                        else:
                            logger.info(f'GPU-ID {gpu_id}: iter {iter}, descendant {i}: pass_call {pass_call}, pass_exe {pass_exe}, speedup {speedup}')
                        
                        record_path = os.path.join(self.repo_path, f'eval.eval_record_iter{iter}_des{i}.json')
                        with open(record_path, "w") as f:
                            cur_data = {"stdout":stdout, "stderr":stderr}
                            f.write(json.dumps(cur_data, ensure_ascii=False)+'\n')
                    except Exception as e:
                        logger.info(f"failed to test the code for {self.memories.ps.file_path} due to {e}")
                        #logger.info(result)
                        raw_code.test_stdout = f"failed to test the code due to: {result.stdout}"
                        raw_code.test_stderr = f"failed to test the code due to: {result.stderr}"
                        continue
                    if not pass_call:
                        raw_code.test_stdout = stdout
                        raw_code.test_stderr = stderr
                        raw_code.profiling = None
                    elif pass_call and not pass_exe:
                        raw_code.pass_call = True
                        raw_code.test_stdout = stdout
                        raw_code.test_stderr = stderr if stderr else stdout
                        self.memories.call_candidate = raw_code.code
                        self.memories.temp_strategy = raw_code.strategy
                        self.memories.pass_call = True
                        raw_code.profiling= None
                    else:
                        raw_code.pass_call = True
                        raw_code.pass_exe = True
                        self.memories.pass_call = True
                        self.memories.exe_candidate = raw_code.code
                        self.memories.call_candidate = raw_code.code
                        self.memories.temp_strategy = raw_code.strategy
                        raw_code.profiling= None
                    self.memories.call_err_msg = raw_code.test_stdout
                    self.memories.exe_err_msg = raw_code.test_stderr
                    if speedup > 0.0 and pass_exe:
                        raw_code.pass_perf = True
                        self.memories.pass_perf = True
                        raw_code.metric = speedup
                        raw_code.latency = latency
                    else:
                        raw_code.metric = 0.0
                    
                self.config.descendant_debug = min(self.config.descendant_debug,len(self.memories.raw_codes))
                if sum(rc.pass_exe for rc in self.memories.raw_codes)>=self.config.descendant_debug:
                    self.memories.pass_exe = True

            # generate reflections
            if self.forbid_reflection:
                logger.info("FORBID REFLEXION")
            elif iter == self.config.iteration_num - 1:
                logger.info(f"\ndo not need to generate reflections in the last iter")
            else:
                logger.info(f"\ngenerate reflections")
                self.generate_reflexion(self.memories)
           

            # update perf_candidates
            logger.info(f"\nupdate perf_candidates")
            if self.memories.raw_codes:
                for i in range(len(self.memories.raw_codes)):
                    raw_code = self.memories.raw_codes[i]
                    self.memories.history[i].append(raw_code)
                    codes_sorted = sorted(self.memories.history[i], key=lambda x: x.metric, reverse=True)
                    self.memories.history[i] = codes_sorted[:5]
                    if raw_code.pass_perf:
                        self.update_perf_candidates(mem=self.memories, raw_code=raw_code, ancestor_num=self.config.ancestor_num, metric_order=self.config.metric_order)
            for i in range(len(self.memories.perf_candidates)):
                logger.info("Candidate {} perf {} speedup {}".format(i+1, self.memories.perf_candidates[i][1], self.memories.perf_candidates[i][2]))
            if len(self.memories.perf_candidates) > 0:
                self.memories.ps.solution = self.memories.perf_candidates[0][0]
                self.memories.ps.speedup = self.memories.perf_candidates[0][2]
            elif self.memories.exe_candidate:
                self.memories.ps.solution = self.memories.exe_candidate
            elif self.memories.call_candidate:
                self.memories.ps.solution = self.memories.call_candidate
            elif self.memories.raw_codes:
                self.memories.ps.solution = self.memories.raw_codes[0].code

            if self.config.output_path is not None:
                self.write_memories(mem_output_path)


        if len(self.memories.perf_candidates) > 0:
            self.memories.ps.label = self.memories.perf_candidates[0][0]
            return self.memories.ps.label, self.memories.perf_candidates[0][2]

        else:
            self.memories.ps.label = self.memories.ps.ori_full_code
            logger.info('[WARNING] OURLLM optimization failed, fall back to ori code')
            return self.memories.ps.label, 1.0

    
    def generate_solution(self, mem, iter):

        text = mem.ps.instruction

        # for the one that has perf_candidates, and the code generated in this round pass_exe, we need to generate a new code
        # for the one that has perf_candidates, but the code generated in this round not pass_exe, if the debug_num has exceeds the man_debug_num, then generate a new code
        # otherwise, go to debug
        if (mem.perf_debug_num >= self.config.max_perf_debug_num) or mem.pass_exe:
            mem.perf_debug_num = 0
            mem.raw_codes = None
        
        
        label_kernel = mem.ps.kernel_code

        prompt_cand_num = 0
        prompt_trial_num = 0
        if len(mem.perf_candidates) > 0 and not mem.raw_codes:
            text += f"\nHere is an example snippet of baseline code that runs in low latency but correctly validated: {label_kernel}"
            text += """\nThere are some reference codes(NO.1, NO.2 and so on). According to their performance(latency in ms) and the corresponding analysis, you need to continue optimize the code with better performance. You should maintain code correctness during optimization."""
            text +="\nYou can use optimization strategies such as Memory access efficiency, Hardware resource utilization, IR analysis, Assembly analysis, Kernel occupancy."

            for i, cand in enumerate(mem.perf_candidates):
                #text += f"\n### Reference {i+1}"
                #text += f"\nOptimized code: {cand[0]}"
                #print(cand)
                last_full_code = cand[0]
                last_kernel_code, _ = extract_hip_kernels(replace_defines_forward(last_full_code, define_dict=mem.ps.define_dict), mem.ps.kernel_func_name, init=False)
                text += f"\nreference code No.{i}: {last_kernel_code}"
               
                if type(cand[1]) is list and len(cand[1]) <= 5 and len(cand[1]) > 1:
                    text += f"\nreference code latency(e.g. ms) (multiple latency number corresponds to different input or forward/backward in order if implemented): {cand[1]}"
                elif len(cand[1]) == 1:
                    text += f"\nreference code latency(e.g. ms): {cand[1][0]}"
                
                
                text += f"\nreference code latency ratio to original baseline: {self.ori_speedup / cand[2]}"
                if cand[3]:
                    text += f"\nreference code Analysis: {cand[3]}"
                prompt_cand_num += 1
            text += "\nAnalyze and compare all reference code, based on these reference code/perf/analysis and give a optimized version of code motivated by them."
            text += "\nThink before writing the optimization and no more explanation is required after the thinking."
            text += "\nYou should not suggest changes to the name of the function and parameter names, counts, or order."
        #if mem.raw_codes and len(mem.perf_candidates) < self.config.ancestor_num:
        else:
            if not mem.raw_codes or mem.raw_codes[0] == "":
                text += f"\nHere is an example snippet of baseline code: {label_kernel}"
            else:
                pre_attempt = 1
                text += f"\nHere is an example snippet of baseline code: {label_kernel}"
                for i in range(len(mem.raw_codes)):
                
                    raw_code = mem.raw_codes[i]
                    if raw_code.pass_perf:
                        continue
                    
                    pre_attempt_kernel, _ = extract_hip_kernels(replace_defines_forward(raw_code.code, mem.ps.define_dict), mem.ps.kernel_func_name, init=False) 
                    text += f"\nPrevious attempt implementation {str(pre_attempt)}:{pre_attempt_kernel}"

                    if not raw_code.pass_call:
                        text += f"\nTest messages for this previous attempt:{raw_code.test_stdout}"
                        text += f"\nTest messages for correctness check of this previous attempt:{raw_code.test_stderr}"
                
                    elif not raw_code.pass_exe:
                        text += "\nThis previous attempt implementation can be run successfully."
                        text += f"\nTest messages for correctness check of this previous attempt:{raw_code.test_stderr}"

                    if raw_code.reflections:
                        text += f"\nReflection on this previous attempt:{raw_code.reflections}"
                    pre_attempt += 1
                    prompt_trial_num += 1
                    #if prompt_trial_num >= self.config.ancestor_num - len(mem.perf_candidates):
                    #    break
        logger.info(f"GPU-ID {self.gpu_id}: ====== GOT {prompt_cand_num} candidates and {prompt_trial_num} failed trials in this iteration ======")

        text += "\nOutput your answer in json format, with the format as follows: {\"thought\": \"\", \"code\": \"\"}. Please strictly output in JSON format."
        text += "\nGenerate the correct and optimized code without explanation, which we can run directly in the \"code\" field."
        
        gens_codes: List[tempCode] = []
        for i in range(self.config.descendant_num):
            gen_code = tempCode()
            try:
                gen_code.code, gen_code.strategy = self.call_llm_code(prompt=text, temperature=self.config.temperature, mem=mem, seed=i, iter=iter)
            except Exception as e:
                logger.info(f"failed to call LLM for {mem.ps.file_path} due to {e}")
            gens_codes.append(gen_code)
       
        mem.raw_codes = gens_codes
        mem.pass_exe = False
        mem.pass_call = False
        mem.pass_perf = False
        return
    
    
    def generate_reflexion(self, mem):
        if mem.raw_codes :
            for i in range(len(mem.raw_codes)):
                raw_code = mem.raw_codes[i]
                if  raw_code.reflections:
                    continue
                if raw_code.pass_perf:
                    reflect_txt = prompt_for_reflection_hip.prompt_ga_hip.format(
                        problem=mem.ps.instruction,
                        original_code=mem.ps.ori_full_code, #replace_defines_forward(mem.ps.ori_full_code, mem.ps.define_dict),
                        code=raw_code.code,
                        latency=raw_code.latency,
                        latency_ratio=1/raw_code.metric,
                        exe_test_result=raw_code.test_stderr
                    )
                elif raw_code.pass_call:
                    reflect_txt = prompt_for_reflection_hip.prompt_exe_hip.format(
                        problem=mem.ps.instruction,
                        original_code=mem.ps.ori_full_code, #replace_defines_forward(mem.ps.ori_full_code, mem.ps.define_dict),
                        solution=raw_code.code,
                        call_test_result="succeed",
                        exe_test_result=raw_code.test_stderr
                    )
                else:
                    reflect_txt = prompt_for_reflection_hip.prompt_hip.format(
                        problem=mem.ps.instruction,
                        original_code=mem.ps.ori_full_code, #replace_defines_forward(mem.ps.ori_full_code, mem.ps.define_dict),
                        solution=raw_code.code,
                        test_result=raw_code.test_stdout
                    )
                reflect_msg = [
                    {
                        "role": "user",
                        "content": reflect_txt
                    }
                ]
                try:
                    raw_code.reflections = self.model.generate(reflect_msg)
                except:
                    raw_code.reflections = None

    
    def call_llm_code(self, prompt, temperature, mem, seed, iter):
        if self.use_ori_code:
            logger.info("IN DEBUG MODE: USE_ORI_CODE, output the original code directly")
            return mem.ps.ori_full_code, ""
        
        msg = [{"role": "user", "content": prompt}]
       
        response = self.generator.generate(msg, temperature=1.0, max_tokens=40960, seed=seed)
        record_path = os.path.join(self.repo_path, f'trial.gen_record_iter{iter}_des{seed}.json')
        with open(record_path, "w") as f:
            cur_data = {"prompt":msg[0]["content"], "response":response}
            f.write(json.dumps(cur_data, ensure_ascii=False)+'\n')
       
        try:
            code = clear_json(response)["code"]
            #code = mem.ps.kernel_code
            strategy = clear_json(response)['thought']
        except:
            logger.info(f"failed to extract code for {mem.ps.kernel_func_name}")
            try:
                code = response.split("\"code\":")[1]
                code = code.split("}")[0]
                code = clear_code(code)
                strategy = ""
            except:
                code = None
                strategy = ""
        
        if code is None:
            logger.info(f"raw code for {mem.ps.kernel_func_name} is None")
            code = ""
            strategy = ""
        else:
            label_kernel= mem.ps.kernel_code
            dist = dtw_string_distance(code.split('\n'),label_kernel.split('\n'))
            # logger.info("label kernel")
            # logger.info(label_kernel)
            # logger.info('~'*50)
            # logger.info("gen code")
            # logger.info(code)
            
            
            #logger.info(f'the dtw dist of generated kernel is {dist}')
            if dist < self.dtw_threshold:
                ori_raw_code = code
                try:
                    for _ in range(self.max_trial_num):
                        response = self.model.generate(msg, temperature=temperature, max_tokens=self.model.max_length, seed=seed+1+_)
                        code = clear_json(response)["code"]
                        strategy = clear_json(response)['thought']
                        dist = dtw_string_distance(code.split('\n'), label_kernel.split('\n'))
                        #logger.info(f'got duplicate, the regenerated dtw dist of generated kernel is {dist}')
                        if dist > self.dtw_threshold:
                            break
                except:
                    code = ori_raw_code
            new_kernel_body = extract_kernel_body(code)
            #logger.info(f"starting to extract and replace kernel body for {mem.ps.kernel_func_name}")
            #logger.info('~'*50)
            #logger.info("new kernel ")
            #logger.info(code)
            code = replace_function_body(replace_defines_forward(mem.ps.ori_full_code, mem.ps.define_dict), mem.ps.kernel_func_name, mem.ps.kernel_code, new_kernel_body)
            # logger.info('~'*50)
            # logger.info("full code")
            # logger.info(code)
            # replace back to defines
            code = replace_defines_backward(code, mem.ps.define_dict)
            #dtw_dist = dtw_string_distance(code.split('\n'), mem.ps.ori_full_code.split('\n'))
            #logger.info(f'The solution full code got dtw dist from the ori code: {dtw_dist}')

        return code, strategy 




    def update_perf_candidates(self,mem, raw_code: tempCode, ancestor_num=5, metric_order=True):
        if len(mem.perf_candidates) < ancestor_num:
            candidate = [raw_code.code, raw_code.latency, raw_code.metric,  raw_code.reflections, raw_code.profiling]
            mem.perf_candidates.append(tuple(candidate))
            mem.perf_candidates = sorted(mem.perf_candidates, key=lambda x: x[2], reverse=metric_order)


        elif (metric_order and mem.perf_candidates[-1][2] <= raw_code.metric) or (not metric_order and mem.perf_candidates[-1][2] > raw_code.metric):
            candidate = [raw_code.code, raw_code.latency, raw_code.metric, raw_code.reflections, raw_code.profiling]
            mem.perf_candidates[-1] = tuple(candidate)
            # order the candidates in ascending order with regard to speedups
            mem.perf_candidates = sorted(mem.perf_candidates, key=lambda x: x[2], reverse=metric_order)
       
    