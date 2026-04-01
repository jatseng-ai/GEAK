export KERNEL_LLM_VISIBLE_DEVICES=6,7

TASK_NAME="device_search_n"
#TASK_NAME="silu"
rm -rf /workspace
mkdir /workspace
cp -r ${TASK_NAME} /workspace
python3 /group/ossdphi_algo_scratch_16/cohuang/260323_GEAK_v3/src/minisweagent/tools/geak_kernel_llm.py

