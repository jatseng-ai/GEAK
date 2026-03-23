from agents.GaAgent import GaAgent
from models.OpenAI import OpenAIModel
from args_config import load_config


def main():
    args = load_config("configs/tritonbench_gaagent_config.yaml")

    # setup LLM model
    model = OpenAIModel(api_key=args.api_key, model_id=args.model_id)

    # setup agent
    agent = GaAgent(model,
                    max_perf_debug_num=args.max_perf_debug_num, 
                    mem_file=args.mem_file, 
                    descendant_num=args.descendant_num, 
                    output_path=args.output_path, 
                    iteration_num=args.iteration_num, 
                    temperature=args.temperature, 
                    ancestor_num=args.ancestor_num, 
                    mutation=args.mutation, 
                    gpu_id=args.gpu_id, 
                    descendant_debug=args.descendant_debug, 
                    metric_order=args.metric_order,
                    repo_dir=args.repo_dir)

    # run the agent
    agent.run(code_path=args.code_path, 
              prompt=args.prompt, 
              test_case=args.test_case)


if __name__ == "__main__":
    main()