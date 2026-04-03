from tqdm import tqdm
import os
import json
import yaml
from typing import List
import subprocess
from loguru import logger
import sys

from geak_agent.agents.reflexion_oneshot import Reflexion_Oneshot
from geak_agent.utils.utils import clear_code, clear_json, extract_json_from_stdout
from geak_agent.memories.Memory import MemoryClassMeta
from geak_agent.prompts import prompt_for_generation, prompt_for_reflection
from geak_agent.dataloaders.ProblemState import tempCode, ProblemState
from geak_agent.models.OpenAI import OpenAIModel
from geak_agent.models.Claude import ClaudeModel
from geak_agent.models.Gemini import GeminiModel

class GaAgent(Reflexion_Oneshot):
    def __init__(self, config_path):
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
                                                             "history"]):
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
                            history=[[] for _ in range(descendant_num)]
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
                history=[[] for _ in range(descendant_num)]
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
                "pass_perf": self.memories.pass_perf
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

    def run(self, code_path=None, prompt=None, test_case=None):
        """
        Args:
            output_path: the folder to store the final result
            multi_thread: whether use multithreading for generating
            datalen: for debug, to specify how many data from the dataset you want to use
            iteration_num: how many iterations you want to run
            temperature: LLM temperature
            ancestor_num: how many samples you want to add in the prompt when optimize the code
            descendant_num: how many codes you want to generate in one try
            start_idx: start idx of the data rows
            gpu_id: which gpu you want to use when you test the scripts
            start_iter: which iteration you want to start with. useful when you load previous result and memory
        """
        assert self.config.ancestor_num >= 0, f"expect ancestor_num to be larger than 0, but got {self.config.ancestor_num}"
        assert self.config.descendant_num >= 0, f"expect descendant_num to be larger than 0, but got {self.config.descendant_num}"
        assert self.config.descendant_debug >= 0, f"expect descendant_debug to be larger than 0, but got {self.config.descendant_debug}"
        assert prompt is not None, f"expect prompt, but got None"
        assert test_case is not None, f"expect test case, but got None"
        if self.config.reset_repo:
            subprocess.run([f"git config --global --add safe.directory {self.config.repo_dir}"], shell=True, cwd=self.config.repo_dir, capture_output=True, text=True, timeout=20)
            subprocess.run(["git restore ."], shell=True, cwd=self.config.repo_dir, capture_output=True, text=True, timeout=20)
        if code_path and os.path.exists(code_path):
            with open(code_path, "r", encoding="utf-8") as f:
                self.memories.ps.label = f.read()
        self.memories.ps.file_path = code_path
        self.memories.ps.test_code = test_case
        with open(prompt, "r", encoding="utf-8") as f:
            prompt_config = yaml.safe_load(f)
        self.memories.ps.instruction = prompt_config["instance_template"]
        self.memories.ps.instruction += self.related_file()
        self.memories.ps.system = prompt_config["system_template"]
        logger.info(f"\n===GPT-5 Optimize {code_path} ===")
        for iter in range(self.config.iteration_num):
            logger.info(f"\n===GPT-5 Iteration {iter} ===")
            if self.config.output_path is not None:
                os.makedirs(os.path.dirname(self.config.output_path), exist_ok=True)
                root, extension = os.path.splitext(self.config.output_path)
                iter_path = f"{root}_{iter}{extension}"
                mem_output_path = f"{root}_mem_{iter}.json"
            # import pdb; pdb.set_trace()
            # generate solution
            logger.info(f"\ngenerate solution")
            self.generate_solution(self.memories)
            
            # generate LLM evaluation
            logger.info(f"\ngenerate LLM evaluation")
            self.generate_llm_evaluate(self.memories)
            
            # run scripts
            logger.info(f"\nrun scripts on gpu")
            # import pdb; pdb.set_trace()
            if self.config.output_path is not None:
                root, extension = os.path.splitext(self.config.output_path)
                tmp_dir = f"{root}_tmp"
                exe_dir = f"{root}_pass_exe"
                perf_result_dir = f"{root}_perf_results"
                perf_log_dir = f"{root}_perf_logs"

            else:
                tmp_dir = "tmp"
                exe_dir = "pass_exe"
                perf_result_dir = "perf_results"
                perf_log_dir = "perf_logs"
            if self.memories.raw_codes:
                for i in range(len(self.memories.raw_codes)):
                    raw_code = self.memories.raw_codes[i]
                    speedup = 0.0
                    if raw_code.pass_perf:
                        continue
                    try:
                        if raw_code.code :
                            with open(code_path, "w", encoding="utf-8") as f:
                                f.write(raw_code.code)
                            # import pdb; pdb.set_trace()
                            command = [f"HIP_VISIBLE_DEVICES={self.config.gpu_id} python {self.memories.ps.test_code}"]
                            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=3600)
                            if result.returncode == 0:
                                output = result.stdout.strip()
                                pass_call = "Failed to run correctness" not in output
                                pass_exe = pass_call and "Failed to run performance" not in output
                                if pass_exe:
                                    speedup = extract_json_from_stdout(output)[-1].get("speedup", 0)
                                stdout, stderr = result.stdout, result.stderr
                            else:
                                pass_call, pass_exe, speedup, stdout, stderr = False, False, 0.0, result.stdout, result.stderr
                        else:
                            pass_call, pass_exe, speedup, stdout, stderr = False, False, 0.0, "", "Code is empty"
                        
                    except Exception as e:
                        print(f"failed to test the code for {self.memories.ps.file_path}")
                        raw_code.test_stdout = f"failed to test the code due to: {e}"
                        raw_code.test_stderr = f"failed to test the code due to: {e}"
                        continue
                    if not pass_call:
                        raw_code.test_stdout = stdout
                        raw_code.test_stderr = stderr
                        raw_code.profilig = None
                    elif pass_call and not pass_exe:
                        raw_code.pass_call = True
                        raw_code.test_stdout = stdout
                        raw_code.test_stderr = stderr if stderr else stdout
                        self.memories.call_candidate = raw_code.code
                        self.memories.temp_strategy = raw_code.strategy
                        self.memories.pass_call = True
                        raw_code.profilig= None
                    else:
                        raw_code.pass_call = True
                        raw_code.pass_exe = True
                        self.memories.pass_call = True
                        self.memories.exe_candidate = raw_code.code
                        self.memories.call_candidate = raw_code.code
                        self.memories.temp_strategy = raw_code.strategy
                        raw_code.profilig= None
                    self.memories.call_err_msg = raw_code.test_stdout
                    self.memories.exe_err_msg = raw_code.test_stderr
                    if speedup > 0.0 and pass_exe:
                        raw_code.pass_perf = True
                        self.memories.pass_perf = True
                        raw_code.metric = speedup
                        raw_code.eff = 0.0
                self.config.descendant_debug = min(self.config.descendant_debug,len(self.memories.raw_codes))
                if sum(rc.pass_exe for rc in self.memories.raw_codes)>=self.config.descendant_debug:
                    self.memories.pass_exe = True

            # generate reflections
            logger.info(f"\ngenerate reflections")
            self.generate_reflexion(self.memories)

            # update perf_candidates

            if self.memories.raw_codes:
                for i in range(len(self.memories.raw_codes)):
                    raw_code = self.memories.raw_codes[i]
                    self.memories.history[i].append(raw_code)
                    codes_sorted = sorted(self.memories.history[i], key=lambda x: x.llm_metric, reverse=True)
                    self.memories.history[i] = codes_sorted[:5]
                    if raw_code.pass_perf and raw_code.strategy:
                        raw_code.strategy = None
                        self.update_perf_candidates(mem=self.memories, raw_code=raw_code, ancestor_num=self.config.ancestor_num, metric_order=self.config.metric_order)
            if len(self.memories.perf_candidates) > 0:
                self.memories.ps.solution = self.memories.perf_candidates[0][0]
                self.memories.ps.speedup = self.memories.perf_candidates[0][1]
            elif self.memories.exe_candidate:
                self.memories.ps.solution = self.memories.exe_candidate
            elif self.memories.call_candidate:
                self.memories.ps.solution = self.memories.call_candidate
            elif self.memories.raw_codes:
                self.memories.ps.solution = self.memories.raw_codes[0].code

            if self.config.output_path is not None:
                self.write_memories(mem_output_path)

            os.system(f'rm -rf {exe_dir}')
            os.system(f'rm -rf {perf_result_dir}')
            os.system(f'rm -rf {perf_log_dir}')
            os.system(f'rm -rf {tmp_dir}')
        if len(self.memories.perf_candidates) > 0:
            self.memories.ps.label = self.memories.perf_candidates[0][0]
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(self.memories.ps.label)
    
    def generate_solution(self, mem):
        # import pdb; pdb.set_trace()
        text = mem.ps.system
        text += mem.ps.instruction
        
        # for the one that has perf_candidates, and the code generated in this round pass_exe, we need to generate a new code
        # for the one that has perf_candidates, but the code generated in this round not pass_exe, if the debug_num has exceeds the man_debug_num, then generate a new code
        # otherwise, go to debug
        if (mem.perf_debug_num >= self.config.max_perf_debug_num) or mem.pass_exe:
            mem.perf_debug_num = 0
            mem.raw_codes =None
        if len(mem.perf_candidates) > 0 and not mem.raw_codes:
            text += """\nThere are some Optimized codes(NO.1, NO.2 and so on) to solve the Problem. The Optimized codes are arranged in ascending order based on their performance. According to their performance and the corresponding analysis, you need to generate a new code with better performance. You should maintain code correctness during optimization."""
            for i, cand in enumerate(mem.perf_candidates):
                text += f"\n### Reference {i+1}"
                text += f"\nOptimized code: {cand[0]}"
                text += f"\nOptimized performance: {cand[1]}"
                if cand[3]:
                    text += f"\nStrategy: {cand[3]}"
                if cand[4]:
                    text += f"\nRocprof-compute profiling result:{cand[4]}"
            if self.config.mutation:
                text += "\nGenerate a better strategy completely different from Optimized Implementation. Based on the better strategy generate a better optimization code."
            else:
                text += "\nAnalyze and compare all optimization strategies based on Optimized Implementation codes and give a better strategy motivated by them. Based on the better strategy generate a better optimization code to get a higher speedup."
        if mem.ps.label:
            text += f"\nHere is baseline implement passed correctness and preformance test. Analyze baseline implement, give a better strategy motivated by them and generate a better optimization code:\n{mem.ps.label}"
        
        if mem.raw_codes :
            for i in range(len(mem.raw_codes)):
                raw_code = mem.raw_codes[i]
                if not raw_code.pass_perf:
                    text_temp = text
                    history_text = self._build_history_prompt(mem.history[i])
                    text_temp += f"\nPrevious attempt implementations:{history_text}"
                    text_temp += prompt_for_generation.system_prompt
                    if raw_code.reflections:
                        raw_code.reflections = None
                    try:
                        raw_code.code, raw_code.strategy = self.call_llm_code(prompt=text_temp, temperature=self.config.temperature)
                    except:
                        logger.info(f"failed to call LLM for {mem.ps.file_path}")
            mem.perf_debug_num +=1
            return
        
        gens_codes: List[tempCode] = []
        for i in range(self.config.descendant_num):
            gen_code = tempCode()
            try:
                text_temp = text
                text_temp += prompt_for_generation.system_prompt
                gen_code.code, gen_code.strategy = self.call_llm_code(prompt=text_temp, temperature=self.config.temperature)
            except:
                logger.info(f"failed to call LLM for {mem.ps.file_path}")
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
                raw_code.reflections = self.model.generate(reflect_msg)

    
    def call_llm_code(self, prompt, temperature):
        # import pdb; pdb.set_trace()
        msg = [{"role": "user", "content": prompt}]
        try:
            # import pdb; pdb.set_trace()
            response = self.model.generate(msg, temperature=temperature, max_tokens=30000)
            opti = clear_json(response)
            if 'code' in opti.keys() and  'strategy' in opti.keys():
                code = clear_code(opti['code'])
                strategy = opti['strategy']
                return code, strategy 
        except:
            logger.info(f"failed to call LLM")
            raise ValueError("failed to call LLM")

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
            candidate = [raw_code.code, raw_code.metric, raw_code.eff, raw_code.reflections, raw_code.profilig]
            mem.perf_candidates.append(tuple(candidate))
            mem.perf_candidates = sorted(mem.perf_candidates, key=lambda x: x[1], reverse=metric_order)

        elif (metric_order and mem.perf_candidates[-1][1] <= raw_code.metric) or (not metric_order and mem.perf_candidates[-1][1] > raw_code.metric):
            candidate = [raw_code.code, raw_code.metric, raw_code.eff, raw_code.reflections, raw_code.profilig]
            mem.perf_candidates[-1] = tuple(candidate)
            # order the candidates in ascending order with regard to speedups
            mem.perf_candidates = sorted(mem.perf_candidates, key=lambda x: x[1], reverse=metric_order)
            
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