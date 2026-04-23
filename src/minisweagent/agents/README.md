# Agent implementations

* `optimization_agent.py` - Main kernel-optimization agent (standalone; used by homogeneous + heterogeneous pipelines).
* `default.py` - Base class used by preprocess subagents (`SelectPatchAgent`, `UnitTestAgent`, `ShapeFixerAgent`).
* `parallel_agent.py` - Orchestrator shell that spawns N `OptimizationAgent` workers across a GPU pool.
