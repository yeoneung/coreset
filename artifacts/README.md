# Experiment artifacts (including WSD-style anchor)

This directory contains the minimal learned-experiment artifacts needed for
the covariance-aware warm-start ablation, plus reference-cost files used to
regenerate other reported tables and plots from cached results.

- `data/train.pt`: residual second-moment fitting split (6,804 trajectories)
- `data/test.pt`: held-out source and reverse-sampling evaluation split
- `data/test_ref.pt`: multistart-oracle reference costs for the source ablation
- `data/test_ref_known.pt`: stronger post hoc reference used by the full-budget
  and few-step obstacle-avoidance study (488 values, first-occurrence condition
  order in `data/test.pt`; distinct from the source-ablation reference)
- `data/quad_test_ref_known.pt`: saved quadrotor reference costs used to restore
  the plotted cost scale from the JSON summaries
- `checkpoints/base_nom.pt`: fixed conditional VP score model
- `checkpoints/wsd_diag.pt`: WSD-style conditional Gaussian anchor

The checkpoint `checkpoints/wsd_diag.pt` is the conditional residual
mean/diagonal-variance predictor selected on a condition-disjoint validation
split.

The ablation never retrains or selects the score model. The small WSD-style
network is trained by `code/experiments/train_wsd_anchor.py` and changes only
the intermediate Gaussian source. The source ablation uses the files in this
directory and does not require another workspace or an external checkpoint.
