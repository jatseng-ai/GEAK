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
from minisweagent.tools.geak_agent.models.VLLM import VLLMModel

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
        self.generator = VLLMModel()
        #except:
        #    raise ImportError("VLLM is not implemented")
        self.dtw_threshold = 0.05
        self.max_trial_num = 3
        self.kernel_run_num = 3

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

    def run(self, code_path=None, test_case=None, kernel_func_name=None, define_dict=None):
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
       
        #cody should modify instruction!
        self.memories.ps.instruction = "Please optimize the following HIP kernel/function for better performance on the ROCm platform (MI250 GPU).\n\
    MI250 specs: 208KB LDS per Compute Unit (CU), 64 CUs total.\n\nYou will receive only a single kernel/function from the .hip file.\n\
    You may only modify the function body, but you must output the entire function including its signature.\n\nAllowed:\n\nRewrite or optimize the function body only.\n\n\
    Add local variables, shared memory, unrolling, vectorized I/O, etc.\n\nReorder code inside the function.\n\nAdd comments inside the function.\n\nNot Allowed:\n\nDo NOT change the function name.\n\n\
    Do NOT change the function signature or parameter types.\n\nDo NOT add, remove, or modify any code outside this function.\n\nNo helper functions\n\nNo new includes\n\nNo new kernels\n\n\
    No changes to launch configuration\n\nDo NOT assume access to any code outside this function.\n\nOptimization guidelines (apply those that fit):\n\nChunked/tiled processing using registers or LDS\n\n\
    Shared-memory buffering (LDS)\n\nDelayed stores to shared memory\n\nVectorized loads/stores (float2/float4/uint4/etc.)\n\nLoop unrolling\n\nBound checks for variable sizes\n\nMinimize warp/wavefront divergence\n\n\
    Increase ILP via interleaving independent ops\n\nReduce LDS/register usage for higher occupancy\n\nFavor coalesced memory and AMD wavefront-friendly access patterns\n\nFuse operations where possible\n\n\
    Use compiler hints like #pragma unroll\n\nHard Requirements:\n\nReturn the full function, including the exact original function signature.\n\nOnly modify code inside the function body.\n\n\
    Preserve algorithmic correctness and bitwise-equivalent outputs.\n\nMaintains existing formatting and comments unless improving them.\n\nCode must be compilable and runnable."

        # cody, currently not use system prompt right now, to be accord with AIG-Eval
        self.memories.ps.system = """
  You are an expert C++ programmer specializing in AMD ROCm kernels.
  **Performance Analysis Guidelines**
  When analyzing performance bottlenecks, consider:
  1. **GPU Occupancy Analysis**:
    - Check if the number of launched blocks is sufficient for good GPU utilization
    - For few segments (e.g., ≤32), single-block-per-segment may lead to low occupancy
    - Consider multi-block parallelization strategies when segment count is small
  2. **Architecture-level Optimizations**:
    - Beyond code-level optimizations (loop unrolling, memory access), consider:
      - Parallelizing segment processing across multiple blocks
      - Two-phase reduction strategies (partial results + final reduction)
      - Adaptive strategies based on segment count
  3. **Test Scenario Awareness**:
    - Benchmark tests segment counts: 1, 10, 100, 1000, 10000
    - Pay attention to performance across ALL segment counts
    - Low segment counts (1, 10) may require different optimization strategies than high counts (1000, 10000)
"""
        #init test, got the baseline performance first
        logger.info(f"Setting original perf for comparison for {kernel_func_name}...")
        os.system("rm -rf /root/.cache/torch_extensions/")
        if False:
            self.ori_latency = 0
            self.ori_speedup = 0
        else:
            command = [f"{self.memories.ps.test_code}"]
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=3600)
            try:
                output = result.stdout.strip()
                self.ori_speedup = extract_json_from_stdout(output)[-1].get("speedup", 0)
                self.ori_latency = extract_json_from_stdout(output)[-1].get("latency", 0)
            except Exception as e:
                logger.info(f"failed to test the original code for {kernel_func_name}, please check configs or the original code. the reason is: {e}")
                exit()
        logger.info(f"Original latency set successfully! It is {self.ori_latency}")
        logger.info(f"Original speedup set successfully! It is {self.ori_speedup}")

            

        logger.info(f"\n===OURLLM Optimize {code_path} ===")
        for iter in range(self.config.iteration_num):
            logger.info(f"\n===OURLLM Iteration {iter} ===")
            if self.config.output_path is not None:
                os.makedirs(os.path.dirname(self.config.output_path), exist_ok=True)
                root, extension = os.path.splitext(self.config.output_path)
                mem_output_path = f"{root}_mem_{iter}.json"

            # generate solution
            logger.info(f"\ngenerate solution")
            self.generate_solution(self.memories)
            
            # generate LLM evaluation
            #logger.info(f"\ngenerate LLM evaluation")
            #self.generate_llm_evaluate(self.memories)
            
            # run scripts
            logger.info(f"\nrun scripts on gpu")
            if self.memories.raw_codes:
                for i in range(len(self.memories.raw_codes)):
                    raw_code = self.memories.raw_codes[i]
                    speedup = 0.
                    latency = 9999
                    if raw_code.pass_perf:
                        continue
                    try:
                        if raw_code.code:
                            with open(code_path, "w", encoding="utf-8") as f:
                                f.write(raw_code.code)
                            # import pdb; pdb.set_trace()
                            os.system("rm -rf /root/.cache/torch_extensions/")
                            #command = [f"HIP_VISIBLE_DEVICES={self.config.gpu_id} {self.memories.ps.test_code}"]
                            command = [f"{self.memories.ps.test_code}"]
                            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=3600)
                            
                            if result.returncode == 0:
                                output = result.stdout.strip()
                                pass_call = "Failed to run correctness" not in output
                                pass_exe = pass_call and "Failed to run performance" not in output
                                if pass_exe:
                                    speedup = extract_json_from_stdout(output)[-1].get("speedup", 0)   
                                    latency = extract_json_from_stdout(output)[-1].get("latency", 0)                                  
                                stdout, stderr = result.stdout, result.stderr
                            else:
                                pass_call, pass_exe, speedup, latency, stdout, stderr = False, False, 0.0, 9999,  result.stdout, result.stderr
                        else :
                            pass_call, pass_exe, speedup, latency, stdout, stderr = False, False, 0.0, 9999, "", "Code is empty"
                        if len(latency) < 5:
                            logger.info(f'iter {iter}, descendant {i}: pass_call {pass_call}, pass_exe {pass_exe}, latency {latency}, speedup {speedup}')
                        else:
                            logger.info(f'iter {iter}, descendant {i}: pass_call {pass_call}, pass_exe {pass_exe}, speedup {speedup}')
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
            #logger.info(f"\ngenerate reflections")
            #self.generate_reflexion(self.memories)
           

            # update perf_candidates
            logger.info(f"\nupdate perf_candidates")
            if self.memories.raw_codes:
                for i in range(len(self.memories.raw_codes)):
                    raw_code = self.memories.raw_codes[i]
                    self.memories.history[i].append(raw_code)
                    #codes_sorted = sorted(self.memories.history[i], key=lambda x: x.llm_metric, reverse=True)
                    codes_sorted = sorted(self.memories.history[i], key=lambda x: x.metric, reverse=True)
                    self.memories.history[i] = codes_sorted[:5]
                    if raw_code.pass_perf: #and raw_code.strategy:
                        #raw_code.strategy = None
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
        else:
            self.memories.ps.label = self.memories.ps.ori_full_code
            logger.info('[WARNING] OURLLM optimization failed, fall back to ori code')
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(self.memories.ps.label)
    
    def generate_solution(self, mem):
        # import pdb; pdb.set_trace()
        #text = mem.ps.system
        text = mem.ps.instruction
        
        # for the one that has perf_candidates, and the code generated in this round pass_exe, we need to generate a new code
        # for the one that has perf_candidates, but the code generated in this round not pass_exe, if the debug_num has exceeds the man_debug_num, then generate a new code
        # otherwise, go to debug
        if (mem.perf_debug_num >= self.config.max_perf_debug_num) or mem.pass_exe:
            mem.perf_debug_num = 0
            mem.raw_codes =None
        
        
        label_kernel = mem.ps.kernel_code
        text += f"\nHere is an example snippet of baseline code that runs in low latency but correctly validated: {label_kernel}"
        if type(self.ori_latency) is list and len(self.ori_latency) <=5:
            text += f"\nbaseline code latency(e.g. ms) (multiple latency number corresponds to different input or forward/backward in order if implemented): {self.ori_latency}"
        else:
            text += f"\nbaseline code latency(e.g. ms): {self.ori_latency}"
        
        prompt_cand_num = 0
        prompt_trial_num = 0
        if len(mem.perf_candidates) > 0 and not mem.raw_codes:
           
            text += """\nThere are some reference codes(NO.1, NO.2 and so on). According to their performance(latency in ms) and the corresponding analysis, you need to continue optimize the code with better performance. You should maintain code correctness during optimization."""
            text +="\nYou can use optimization strategies such as Memory access efficiency, Hardware resource utilization, IR analysis, Assembly analysis, Kernel occupancy."

            for i, cand in enumerate(mem.perf_candidates):
                text += f"\n### Reference {i+1}"
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
                text += f"\nreference code Analysis: {cand[3]}"
                prompt_cand_num += 1
            text += "\nAnalyze and compare all reference code, based on these reference code/perf/analysis and give a optimized version of code motivated by them."
           
        if mem.raw_codes and len(mem.perf_candidates) < self.config.ancestor_num:
            pre_attempt = 1
            for i in range(len(mem.raw_codes)):
               
                raw_code = mem.raw_codes[i]
                if raw_code.pass_perf:
                    continue
                
                pre_attempt_kernel, _ = extract_hip_kernels(replace_defines_forward(raw_code.code, mem.ps.define_dict), mem.ps.kernel_func_name) 
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
                if prompt_trial_num >= self.config.ancestor_num - len(mem.perf_candidates):
                    break
        logger.info(f"====== GOT {prompt_cand_num} candidates and {prompt_trial_num} failed trials in this iteration ======")

       
        text += "\nThink before writing the optimization and no more explanation is required after the thinking."
        text += "\nYou should not suggest changes to the name of the function and parameter names, counts, or order."
        text += "\nOutput your answer in json format, with the format as follows: {\"thought\": \"\", \"code\": \"\"}. Please strictly output in JSON format."
        gens_codes: List[tempCode] = []
        for i in range(self.config.descendant_num):
            gen_code = tempCode()
            try:
                #if i == 0:
                #    logger.info(text)
                gen_code.code, gen_code.strategy = self.call_llm_code(prompt=text, temperature=self.config.temperature, mem=mem, seed=i)

                #gen_code.code, gen_code.strategy = mem.ps.ori_full_code, ""
                #logger.info(f"{gen_code.code} \n\n\n is generated")
            except Exception as e:
                logger.info(f"failed to call LLM for {mem.ps.file_path} due to {e}")
            gens_codes.append(gen_code)
       
        mem.raw_codes = gens_codes
        mem.pass_exe = False
        mem.pass_call = False
        mem.pass_perf = False
        return
    
    
    def generate_reflexion(self, mem):
        
        m_info = """
- runnable test: test if the code can be successfully executed.
- correctness test: test if the output of the code is correct, i.e. if the code does implement the functionality required in the original problem.
- speedup: measures the performance from kernel launch to completion, reflecting the responsiveness and overhead of executing a single instance of the kernel on the GPU. And compare the time with golden reference code to get speedup.
"""
        
        if mem.raw_codes :
            for i in range(len(mem.raw_codes)):
                raw_code = mem.raw_codes[i]
                if  raw_code.reflections:
                    continue
                history_text = self._build_history_prompt(mem.history[i])
                if raw_code.pass_exe:
                    result_txt = f"""
- runnable test: Succeed
- correctness test: Succeed
- speedup: {raw_code.metric}
"""                 
                    reflect_txt = prompt_for_reflection.prompt_evolve_strategy_optimize.format(
                        system=mem.ps.system,
                        instruction=mem.ps.instruction,
                        metrics_info=m_info,
                        evolution_history=history_text,
                        current_program=raw_code.code,
                        test_result=result_txt,
                        reflection=raw_code.reflections
                    )
                else:
                    if raw_code.pass_call:
                        result_txt = f"""
- runnable test: Succeed
- correctness test: Failed
Error Message: {raw_code.test_stderr}
"""
                    else:
                        result_txt = f"""
- runnable test: Failed
Error Message: {raw_code.test_stderr}
- correctness test: Failed
Error Message: {raw_code.test_stderr}
"""

                    reflect_txt = prompt_for_reflection.prompt_evolve_reflect.format(
                        system=mem.ps.system,
                        instruction=mem.ps.instruction,
                        metrics_info=m_info,
                        evolution_history=history_text,
                        current_program=raw_code.code,
                        test_result=result_txt,
                        reflection=raw_code.reflections
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

    
    def call_llm_code(self, prompt, temperature, mem, seed):
        # import pdb; pdb.set_trace()
        msg = [{"role": "user", "content": prompt}]
       
        response = self.generator.generate(msg, temperature=1.0, max_tokens=32768, seed=seed)
        # if response is not None:
        #     record_dir = "record"
        #     record_path = os.path.join(record_dir, mem.ps.filename+f'.gen_record_des_{i}')
        #     os.makedirs(record_dir, exist_ok=True)
        #     with open(record_path, "w") as f:
        #         f.write(response)
        try:
            code = clear_json(response)["code"]
            strategy = clear_json(response)['thought']
        except:
            logger.info(f"failed to extract code for {mem.ps.kernel_func_name}")
            # fail_dir = "failed_to_extract"
            # fail_path = os.path.join(fail_dir, mem.ps.filename+'.gen_fail')
            # os.makedirs(fail_dir, exist_ok=True)

            # if response is not None:
            #     with open(fail_path, "w") as f:
            #         f.write(response)
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
            
            
            logger.info(f'the dtw dist of generated kernel is {dist}')
            if dist < self.dtw_threshold:
                ori_raw_code = code
                try:
                    for _ in range(self.max_trial_num):
                        response = self.model.generate(msg, temperature=temperature, max_tokens=self.model.max_length, seed=seed+1+_)
                        code = clear_json(response)["code"]
                        strategy = clear_json(response)['thought']
                        dist = dtw_string_distance(code.split('\n'), label_kernel.split('\n'))
                        logger.info(f'got duplicate, the regenerated dtw dist of generated kernel is {dist}')
                        if dist > self.dtw_threshold:
                            break
                except:
                    code = ori_raw_code
                
            logger.info(f"starting to extract and replace kernel body for {mem.ps.kernel_func_name}")
            new_kernel_body = extract_kernel_body(code)
            logger.info('~'*50)
            logger.info("new kernel ")
            logger.info(code)
            code = replace_function_body(replace_defines_forward(mem.ps.ori_full_code, mem.ps.define_dict), mem.ps.kernel_func_name, mem.ps.kernel_code, new_kernel_body)
            # logger.info('~'*50)
            # logger.info("full code")
            # logger.info(code)
            # replace back to defines
            code = replace_defines_backward(code, mem.ps.define_dict)
            dtw_dist = dtw_string_distance(code.split('\n'), mem.ps.ori_full_code.split('\n'))
            logger.info(f'The solution full code got dtw dist from the ori code: {dtw_dist}')

        return code, strategy 


    def call_llm_reflecion(self, prompt, temperature):
        msg = [{"role": "user", "content": prompt}]
        try:
            response = self.model.generate(msg, temperature=temperature, max_tokens=30000)
            opti = clear_json(response)
            if 'reflection' in opti.keys():
                reflection = opti['reflection']
                return reflection 
        except:
            logger.info(f"failed to call LLM")
            raise ValueError("failed to call LLM")

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
       

    def _build_history_prompt(self, history):
        text = ""
        history_template = """
### Attempt {attempt_number}
- Code: 
```
{code}
```

- Test Results: 
{test_results}


- Analysis:
{reflection}
"""
        for i, raw_code in enumerate(history):
            if raw_code.pass_perf:
                test_txt = """
runnable test: Succeed
correctness test: Succeed
speedup: {speedup}
"""
                test_txt = test_txt.format(
                    speedup=raw_code.metric
                )
            elif raw_code.pass_exe and not raw_code.pass_perf:
                test_txt = """
runnable test: Succeed
correctness test: Succeed
"""
            elif raw_code.pass_call and not raw_code.pass_exe:
                test_txt = """
runnable test: Succeed
correctness test: {err_msg}
"""
                test_txt = test_txt.format(
                    err_msg=raw_code.test_stderr
                )
            elif not raw_code.pass_call:
                test_txt = """
runnable test: {err_msg}
"""
                test_txt = test_txt.format(
                    err_msg=raw_code.test_stderr
                )
            text += history_template.format(
                attempt_number=i+1,
                code=raw_code.code,
                test_results=test_txt,
                reflection=raw_code.reflections
            )
        
        return text
    
    
    def generate_llm_evaluate(self, mem):
        if mem.raw_codes :
            for i in range(len(mem.raw_codes)):
                raw_code = mem.raw_codes[i]
                if not raw_code.pass_perf:
                    text = ""
                    text += prompt_for_generation.llm_evaluate_prompt.format(current_program=raw_code.code)
                    msg = [{"role": "user", "content": text}]
                    try:
                        response = self.model.generate(msg, temperature=self.config.temperature)
                        llm_eval = clear_json(response)
                        metric = 0.0
                        for k, v in llm_eval.items():
                            if k == "reasoning":
                                continue
                            if isinstance(v, float) or isinstance(v, int):
                                metric += float(v)
                        raw_code.llm_metric = metric
                        raw_code.llm_eval = llm_eval
                    except Exception as e:
                        logger.info(f"failed to generate LLM evaluation {e}")
                        raise ValueError("failed to generate LLM evaluation")