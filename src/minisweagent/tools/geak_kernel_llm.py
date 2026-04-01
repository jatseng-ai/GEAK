"""
FastMCP quickstart example.

Run from the repository root:
    uv run examples/snippets/servers/fastmcp_quickstart.py
"""
from minisweagent.tools.geak_agent.agents.GaAgent_kernel2kernel import GaAgent_kernel2kernel
from minisweagent.tools.geak_agent.utils.utils_ourllm import  extract_hip_kernels, parse_config_defines, replace_defines_forward
from loguru import logger
from pathlib import Path
import os
import multiprocessing

def run_task_on_gpu(gpu_id: int, common_params: dict):
    worker = GaAgent_kernel2kernel(common_params["geak_config"])
    ori_repo_path = common_params["repo_path"]
    if common_params["gpu_ids"].index(gpu_id) != 0:
        new_repo_path = ori_repo_path+f'_gpu_{gpu_id}'
        os.system(f"cp -r {ori_repo_path} {new_repo_path}")
        logger.info(f"cp -r {ori_repo_path} {new_repo_path}")
        optimized_code, optimized_speedup = worker.run(
            gpu_id=gpu_id,
            code_path=common_params["code_path"].replace(ori_repo_path, new_repo_path),
            test_case=common_params["test_case"].replace(ori_repo_path, new_repo_path),
            kernel_func_name=common_params["kernel_func_name"],
            define_dict=common_params["define_dict"],
            gpu_ids=common_params["gpu_ids"],
            repo_path=new_repo_path,
        )
        os.system(f"rm -rf  {new_repo_path}")
        logger.info(f"rm -rf {new_repo_path}")
    else:
        optimized_code, optimized_speedup = worker.run(
            gpu_id=gpu_id,
            code_path=common_params["code_path"],
            test_case=common_params["test_case"],
            kernel_func_name=common_params["kernel_func_name"],
            define_dict=common_params["define_dict"],
            gpu_ids=common_params["gpu_ids"],
            repo_path=ori_repo_path,
        )

    return {
        "gpu_id": gpu_id,
        "optimized_code": optimized_code,
        "optimized_speedup": optimized_speedup
    }

class GEAK_kernel_llm:
    def __init__(self, config_name="geak_agent/geak_kernel_llm_config.yaml"):
        self.geak_config = str(Path(__file__).parent / config_name)


    # Add an addition tool
    def __call__(self, repo_path:str, code_path: str, test_case: str, kernel_func_name: str, define_config_path: str) -> str: # 
        os.system("rm -rf /root/.cache/torch_extensions/")
        if define_config_path is None or len(define_config_path) <= 1 :
            logger.info(f"Current optimization does not rely on #define configs.")
            define_dict = None
        else:
            try:
                define_dict = parse_config_defines(define_config_path)
                logger.info(f"Found {len(define_dict)} macro definitions:")
                for key, value in define_dict.items():
                    logger.info(f"  {key} -> {value}")
            except:
                raise ImportError(f"The input define config path {define_config_path} is not valid for {kernel_func_name}, please check")


        #gpu_ids = [6,7] 
        gpu_ids = os.environ["KERNEL_LLM_VISIBLE_DEVICES"].split(',')
        print(f"gpu ids {gpu_ids}")
        file_content = open(code_path).read()
        try:
            input_kernel, _ = extract_hip_kernels(replace_defines_forward(file_content, define_dict), kernel_func_name)
            logger.info(f'using our llm to optimize kernel, file full path is {code_path}')
            logger.info(f'All the visible devices are {gpu_ids}, separate into {len(gpu_ids)} parallel tasks')
        except:
            raise ValueError("cannot extract kernel from the given code!")
        
        common_params = {
            "geak_config": self.geak_config,
            "repo_path": repo_path,
            "code_path": code_path,
            "test_case": test_case,
            "kernel_func_name": kernel_func_name,
            "define_dict": define_dict,
            "gpu_ids":gpu_ids
            }
        with multiprocessing.Pool(processes=len(gpu_ids)) as pool:

            tasks = [(gpu, common_params) for gpu in gpu_ids]
            results = pool.starmap(run_task_on_gpu, tasks)

        print("\n" + "="*60)
        print("📦 Collecting all the gpu running results: ")
        print("="*60)

        best_code = ""
        best_speedup = 0
        for res in results:
            print(f"GPU {res['gpu_id']} | partial_code: {res['optimized_code']:100} | speedup: {res['optimized_speedup']}")
            if res['optimized_speedup'] > best_speedup:
                best_code = res['optimized_code']
                best_speedup = res['optimized_speedup']
        
        result = f"geak_kernel_llm optimize {code_path} has finished, got speedup {best_speedup} compared with the input code"
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(best_code)
        return {
        "output": result,
        "returncode": 0
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", type=str, default="silu")
    args = parser.parse_args()
    GEAK = GEAK_kernel_llm(config_name="geak_agent/geak_kernel_llm_config.yaml")

    if args.task_name == "silu":
        repo_path="/workspace/silu"
        code_path="/workspace/silu/silu.hip"
        test_case="cd /workspace/silu && python3 scripts/task_runner.py compile && python3 scripts/task_runner.py correctness && python3 scripts/task_runner.py performance"
        kernel_name="silu_mul_kernel"
        config_path=None
    elif args.task_name == "device_search_n":
        repo_path="/workspace/device_search_n/rocPRIM"
        code_path="/workspace/rocPRIM/rocprim/include/rocprim/device/detail/device_search_n.hpp"
        test_case="cd /workspace/device_search_n && python3 scripts/task_runner.py compile && python3 scripts/task_runner.py correctness && python3 scripts/task_runner.py performance"
        kernel_name="search_n_normal_kernel"
        config_path="/workspace/rocPRIM/rocprim/include/rocprim/config.hpp"
    GEAK(repo_path=repo_path, code_path=code_path, test_case=test_case, kernel_func_name=kernel_name, define_config_path=config_path)