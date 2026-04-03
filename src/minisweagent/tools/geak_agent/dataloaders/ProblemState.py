from typing import List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ProblemState:
    file_path: str
    define_dict: dict
    label: Optional[str] = None
    test_code: Optional[str] = None
    system: Optional[str] = None
    instruction: Optional[str] = None
    solution: Optional[str] = None
    speedup: float = 0.0
    full_code: Optional[str] = None
    kernel_func_name: Optional[str] = None

   

@dataclass
class ProblemStateROCm:
    instruction: str
    label: str
    filename: str
    opname: str
    target_kernel_name: str
    test_code: str
    solution: Optional[str] = None
    pass_call: bool = False
    pass_exe: bool = False
    speedup: float = 0.0

@dataclass
class tempCode:
    code: Optional[str] = None
    strategy: Optional[str] = None
    reflections: Optional[str] = None
    test_stdout: Optional[str] = None
    test_stderr: Optional[str] = None
    profiling: Optional[str] = None
    pass_call: bool = False
    pass_exe: bool = False
    pass_perf: bool = False
    metric: float = 0.0
    latency: float = 0.0
    llm_metric: float = 0.0
    llm_eval: Optional[str] = None
    