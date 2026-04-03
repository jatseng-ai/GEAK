export KERNEL_LLM_VISIBLE_DEVICES=6,7

#TASK_NAME="device_search_n"
TASK_NAME="silu"
rm -rf /workspace
mkdir /workspace
cp -r ${TASK_NAME} /workspace
python3 src/minisweagent/tools/geak_kernel_llm.py --task-name ${TASK_NAME}

