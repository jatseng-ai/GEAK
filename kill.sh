#kill -9 $(ps -ef | grep llamafactory | grep -v grep | awk '{print $2}')
kill -9 $(ps -ef | grep applications | grep -v grep | awk '{print $2}')
#kill -9 $(ps -ef | grep vllm | grep -v grep | awk '{print $2}')
pkill -f 'python3 (main_gaagent_hip_kernel2kernel.py|main.py|task_runner.py)'
pkill -9 -f 'python3 test.*\.py'
pkill -9 -f 'python3 scripts/task_runner.*\.py'
pkill -9 -f 'bash unit_test_geak_kernel_llm.sh'
pkill -9 -f './applications*'
pkill -9 -f '/group/ossdphi_algo_scratch_16'
pkill -9 -f 'python3'
pkill -9 -f '*test*'