# Documentation

Start here:

| page | for |
|---|---|
| [GETTING_STARTED.md](GETTING_STARTED.md) | install, build, first chat, what fits your GPU |
| [STUDIO.md](STUDIO.md) | Forge Studio: every screen, the engines, settings, how it is built |
| [ENGINE.md](ENGINE.md) | `forge serve`: flags, HTTP API, how the engine uses memory, recipes |
| [MODELS.md](MODELS.md) | which models run, what was tested, what does not run |
| [PERFORMANCE.md](PERFORMANCE.md) | measurements against llama.cpp, the 27B on an 8 GB card, what worked and what did not |
| [ROADMAP.md](ROADMAP.md) | what is missing |

The research stack (tokenizer, trainer, cloud jobs):

| page | for |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | from raw text to a served model: data formats, checkpoint format, trainer, engine, JAX port |
| [ENVIRONMENT.md](ENVIRONMENT.md) | every environment variable the code and scripts read |
| [RUNBOOK.md](RUNBOOK.md) | commands to train, fine-tune, evaluate and serve (home GPU, Modal, Kaggle) |
| [KERNEL_TUNING.md](KERNEL_TUNING.md) | notes from tuning the CUDA kernels |
| [windows_build.md](windows_build.md) | building the engine on Windows |
| [SPECS.md](SPECS.md), [TPU_PREMORTEM.md](TPU_PREMORTEM.md), `eval_ladder_validation/` | earlier specifications and validation material |
