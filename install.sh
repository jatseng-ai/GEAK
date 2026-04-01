pip install -e .
pip install loguru
pip install rank_bm25
pip install requests
pip install python-Levenshtein
pip install tenacity
pip install openai
apt-get update
apt-get install -y gawk
apt-get install -y vim
apt-get install -y cmake
apt install -y rocprofiler-compute
update-alternatives --install /usr/bin/rocprof-compute rocprof-compute /opt/rocm/bin/rocprof-compute 0
pip install blinker==1.9 --ignore-installe
python3 -m pip install -r /opt/rocm/libexec/rocprofiler-compute/requirements.txt
#for vllm
pip install setuptools_scm wheel packaging ninja amdsmi

#cd /group/ossdphi_algo_scratch_16/cohuang/vllm/vllm
#git config --global --add safe.directory /group/ossdphi_algo_scratch_16/cohuang/vllm/vllm/.deps/triton_kernels-src
#VLLM_TARGET=rocm pip install -e . --no-build-isolation
