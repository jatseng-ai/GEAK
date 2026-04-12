from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


BASELINES: dict[str, list[float]] = {
    "assign_score_withk": [0.0126, 0.1143, 0.0147, 0.1513, 0.0134, 1.1816, 0.013, 1.2139, 0.013, 4.5081],
    "ball_query": [0.052, 0.0147, 0.1883, 0.1874, 2.8539, 2.8323, 0.2433, 0.0229, 0.3476, 0.3697],
    "furthest_point_sample": [0.0123, 0.0184, 0.0193, 0.0238, 0.0455, 0.0499, 0.1115, 0.1074, 0.2497, 0.2363],
    "gather_points": [0.0117, 0.0114, 0.0114, 0.0115, 0.0116, 0.0116, 0.0117, 0.0113, 0.0113, 0.0117],
    "knn": [
        0.0406,
        0.0475,
        0.0523,
        0.2324,
        0.247,
        0.2876,
        0.5887,
        0.6045,
        0.685,
        1.6793,
        1.6897,
        1.7099,
        0.2433,
        0.2522,
        0.2614,
    ],
    "matrix_multiplication": [394.2077, 397.6859, 416.3669, 380.4248, 394.4292],
    "points_in_boxes": [
        0.0118,
        0.0125,
        0.0123,
        0.0124,
        0.0126,
        0.0125,
        0.0126,
        0.0127,
        0.0129,
        0.0127,
        0.0127,
        0.0128,
        0.0165,
        0.0167,
        0.0155,
        0.0159,
        0.0125,
        0.0124,
        0.0125,
        0.0125,
    ],
    "roiaware_pool3d": [0.076, 0.0757, 0.1896, 0.1873, 0.8102, 0.8094, 0.2891, 0.2889, 0.5519, 0.5515],
    "roipoint_pool3d": [0.0537, 0.1344, 0.9254, 0.4544, 2.4718],
    "silu": [0.0026, 0.0057, 0.0147, 0.0349, 0.0074],
    "three_interpolate": [0.0082, 0.0085, 0.0281, 0.008, 0.0084],
    "three_nn": [0.0183, 0.0585, 0.4436, 0.8749, 0.0318],
}


_PERF_RE = re.compile(r"^\s*Perf(?:ormance)?:\s*([0-9]*\.?[0-9]+)\s*ms\b", re.IGNORECASE)


@dataclass(frozen=True)
class PatchResult:
    path: Path
    avg_speedup: float
    n: int


def _default_logs_dir(repo_root: Path) -> Path:
    for name in ("optimization_logs", "Optimization_logs"):
        p = repo_root / name
        if p.exists():
            return p
    return repo_root / "optimization_logs"


def _parse_perf_ms(path: Path) -> list[float]:
    perfs: list[float] = []
    for line in path.read_text(errors="replace").splitlines():
        m = _PERF_RE.match(line)
        if m:
            perfs.append(float(m.group(1)))
    return perfs


def _avg_speedup(baseline: list[float], perfs: list[float]) -> float:
    return sum(b / t for b, t in zip(baseline, perfs, strict=True)) / len(baseline)


@app.command()
def main(
    kernel: str | None = typer.Argument(None, help="Kernel name (omit to scan all)"),
    top: int = typer.Option(10, "--top", min=1, help="Show top-N patches by avg speedup"),
    logs_dir: Path | None = typer.Option(None, "--logs-dir", exists=False, help="Override optimization_logs directory"),
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    base_dir = logs_dir or _default_logs_dir(repo_root)

    kernels = sorted(BASELINES) if kernel is None else [kernel.strip()]
    if not kernels or any(k not in BASELINES for k in kernels):
        bad = [k for k in kernels if k not in BASELINES]
        raise typer.BadParameter(f"Unknown kernel(s): {', '.join(repr(k) for k in bad)}. Known: {', '.join(sorted(BASELINES))}")

    any_found = False
    for k in kernels:
        baseline = BASELINES[k]
        patch_files = sorted(base_dir.glob(f"{k}*/**/patch_*_test.txt"))
        if not patch_files:
            typer.echo(f"kernel={k}  baseline_n={len(baseline)}  found=0")
            typer.echo("")
            continue

        results: list[PatchResult] = []
        skipped = 0
        for p in patch_files:
            perfs = _parse_perf_ms(p)
            if len(perfs) != len(baseline):
                skipped += 1
                continue
            results.append(PatchResult(path=p, avg_speedup=_avg_speedup(baseline, perfs), n=len(perfs)))

        typer.echo(
            f"kernel={k}  baseline_n={len(baseline)}  found={len(patch_files)}  usable={len(results)}  skipped={skipped}"
        )
        if results:
            any_found = True
            results.sort(key=lambda r: r.avg_speedup, reverse=True)
            typer.echo("rank\tavg_speedup\tpatch_test")
            for i, r in enumerate(results[:top], start=1):
                typer.echo(f"{i}\t{r.avg_speedup:.6f}\t{r.path}")
        typer.echo("")

    if not any_found:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
    # call 
    # python3 rank_speedup.py --top 3 --logs-dir ./optimization_logs/
