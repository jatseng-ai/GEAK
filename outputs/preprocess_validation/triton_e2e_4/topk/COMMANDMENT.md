## SETUP
printf '#!/bin/bash\nexport PYTHONPATH=%s:%s:${PYTHONPATH}\nexport HIP_VISIBLE_DEVICES=%s\nexec python3 "$@"\n' "${GEAK_WORK_DIR}" "${GEAK_REPO_ROOT}" "${GEAK_GPU_DEVICE}" > ${GEAK_WORK_DIR}/run.sh && chmod +x ${GEAK_WORK_DIR}/run.sh

## CORRECTNESS
${GEAK_WORK_DIR}/run.sh ${GEAK_HARNESS} --correctness

## PROFILE
for _i in $(seq 1 2); do ${GEAK_WORK_DIR}/run.sh ${GEAK_HARNESS} --profile > /dev/null 2>&1 || true; done
kernel-profile "${GEAK_WORK_DIR}/run.sh ${GEAK_HARNESS} --profile" --gpu-devices ${GEAK_GPU_DEVICE} --replays 5

## BENCHMARK
${GEAK_WORK_DIR}/run.sh ${GEAK_HARNESS} --full-benchmark ${GEAK_BENCHMARK_EXTRA_ARGS:-}

## FULL_BENCHMARK
${GEAK_WORK_DIR}/run.sh ${GEAK_HARNESS} --full-benchmark ${GEAK_BENCHMARK_EXTRA_ARGS:-}
