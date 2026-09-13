# Code map

`theory_bounds.py` is independent of the learned models and regenerates
`figures/theory_bounds.pdf`. It validates the covariance-aware KL bound,
the exact anchor-bias identity, and the effective-rank behavior used in the
paper.

`nab/` implements the experiments:

- `data.py`, `env*.py`: dataset and task environments
- `diffusion.py`, `model.py`: diffusion schedules and neural models
- `train*.py`, `value.py`: score and value-field training
- `eval*.py`, `mpc.py`: open-loop and receding-horizon evaluation
- `eval_budget.py`: joint evaluation-budget and starting-level sweeps
- `eval_guidance.py`: guidance-weight sweep at a gated noise level
- `eval_reference.py`: two-pass evaluation against a shared cost reference
- `figures*.py`: plots from the JSON summaries in `../results/`
- `plot_style.py`: shared readable labels and below-panel legends
- `mnist.py`: class-conditional image experiment

The pendulum guidance cache and evaluation sweep contain weights
`0, 0.1, 0.25, 0.5, 1, 2, 4`; the manuscript panel displays weights up to 1.
The dotted obstacle-avoidance anchor curve is scaled pooled residual RMS
per trajectory point, not conditional mean error.

The covariance and WSD-style experiment is separated into reusable components:

- `nab/covariance.py`: pooled spectral sources and the conditional diagonal
  Gaussian anchor
- `experiments/train_wsd_anchor.py`: condition-level split and Gaussian-NLL
  training for the conditional mean/variance predictor
- `experiments/covariance_anchor.py`: held-out source and reverse-chain ablation
- `plots/covariance_anchor.py`: manuscript figure and table generation
- `tests/test_covariance.py`: pooled and conditional source numerical checks

From the project root, run `python code/experiments/covariance_anchor.py --seeds 3`
to execute the ablation with the included checkpoints, then
`python code/plots/covariance_anchor.py` to regenerate its figure and table.

Run the focused numerical tests with
`python code/tests/test_covariance.py`.

Run `python code/replot.py` from the repository root to regenerate the
reported empirical figures and tables from included cached outputs. This
also works in the supplementary ZIP and does not invoke model sampling.
The broad plotting entry points also generate `value.pdf` and `frontier.pdf`
as guidance diagnostics in addition to the figures cited in the article.
`python code/tests/test_reporting.py` checks actual score-call counts. The
`nfe` / `n_steps` settings count interpolation intervals; the manuscript
reports the actual calls, including the final clean reconstruction.

The committed results, figures, minimal datasets, fixed score checkpoint, and
small WSD-style anchor checkpoint make this ablation auditable and rerunnable
from this repository alone. Other neural experiments remain reproducible
generated outputs; their entry points and parameters are preserved in the
scripts.

The effective-rank result schema uses `single_axis_family` and `product_family`
for its two synthetic distributions. The WSD training report stores every
epoch's training and validation losses in `training_log`. Run
`python code/tests/test_result_schema.py` to check these cached structures
and their score-call metadata.
