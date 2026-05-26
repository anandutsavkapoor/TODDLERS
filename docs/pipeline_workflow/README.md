# HPC workflow diagram

`hpc_workflow.tex` is a standalone TikZ figure of the TODDLERS HPC campaign: the
dependent SLURM job chain (`shell → unified → dissolved → STAB gate`), the resume
loop, and what one phase job does internally (the worker pool: `srun` → N persistent
workers, each running a modular slice through `run_cloudy_task`).

Build (produces a tightly-cropped PDF), or `\input{hpc_workflow}` into another document:

```bash
pdflatex hpc_workflow.tex
```

Needs only a standard TeX Live install (tikz, amsmath, xcolor).
