# Mini Climate Emulation Challenge

<a target="_blank" href="https://colab.research.google.com/github/WinterSchool2026/ch06-mini-climate-emulation/blob/main/getting_started.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Description

This mini challenge is a first step toward a competitive ClimX submission.

Full Earth System Model simulations are computationally expensive, so fast and scientifically
credible emulators are essential. In this challenge, participants build machine-learning emulators
for daily NORESM2-MM outputs using greenhouse gas and aerosol forcings with historical context.

The focus is not only average-state skill, but also behavior relevant to climate extremes.

## Problem statement

Build a machine learning emulator that maps forcing inputs (greenhouse gases and aerosols),
with historical context, to climate-relevant outputs from NORESM2-MM (climate variables and/or
extreme-event indices).

The emulator should be:
- accurate and robust across scenarios,
- computationally efficient,
- and scalable from the `lite` dataset to the full dataset.

## What you model

- Historical period + future SSP scenarios
- Inputs: GHG forcings and aerosol forcings
- Targets: daily near-surface climate variables
- Evaluation emphasis: high-level extreme-event realism and regional robustness

## Two valid project tracks

### Track A — Pure ML
- Improve architecture, losses, calibration, and efficiency
- Main output: stronger predictive model + ablation evidence

### Track B — Science-driven
- Analyze where/why extremes are missed (region, variable, index)
- Main output: interpretable diagnostics + model-guiding insights

Both tracks are valid and can converge into one submission strategy.

## Suggested 1-week workflow

- Days 1–2: common baseline, setup checks, and diagnostics
- Days 3–4: diverge by track (ML improvements or scientific analysis)
- Day 5: converge into a submission-ready experiment plan

Target outcome by end of week:
- reproducible workflow,
- one concrete model improvement or one scientific finding.

## Key questions

These questions guide model development and evaluation decisions:

1. Machine learning questions.
  - How should we handle non-normal or long-tailed distributions in the input variables?
  - What is the optimal train/ validation split? Should historical data be included in training?
  - How can we scale the same training pipeline from the lite dataset to the full dataset?
2. Scientific questions
  - Which variables and regions contribute most to model error, and can targeted weighting help?
  - How sensitive are results to lead time (near-term vs long-term horizons) across SSP scenarios?
  - Can casual methods improve climate emulators?

## Recommended reading material

**Data and previous challenges**
  - [NorESM2 Models](https://gmd.copernicus.org/articles/13/6165/2020/)
  - [Emulation challenge example: ClimateBench](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021MS002954)
  - [Another emulation challenge example](https://ui.adsabs.harvard.edu/abs/2023AGUFMGC33F1206L/abstract)
  - [Indices used for evaluation](https://climpact-sci.org/indices/)

**Climate emulators**
  [Some nice code for climate emulation](https://github.com/blutjens/climate-emulator-tutorial)
  
**Causality**
- Regularize using environments or anchor variables:
  - [AR6 land regions](https://essd.copernicus.org/articles/12/2959/2020/)
  ```
  import regionmask

  ar6_regions = regionmask.defined_regions.ar6.land
  ```
  - CO2, predicted global mean temperature predictions, or enso?
- Methods
  - Invariant risk minimization [paper](https://arxiv.org/pdf/1907.02893) and code: causal_models/irm.py
  - Anchor regression [old paper](https://arxiv.org/pdf/1801.06229) [newer paper](https://arxiv.org/abs/2403.01865) and [code](https://github.com/homerdurand/anchorMVA)
    ```
    from MVA_algo import ReducedRankRegressor 
    from AnchorOptimalProjector import *

    # Training a RRRR
    anchor = ReducedRankRegressor(rank=1, reg=0.1)
    # Projecting the training data in anchor space
    # take gamma > 1
    AOP = AnchorOptimalProjection(gamma=1.5)
    X_train_transform, Y_train_transform = AOP.fit_transform(A_train_scaled, X_train_scaled, Y_train_scaled)

    # training a anchor model
    anchor.fit(X_train_transform, Y_train_transform)

    # predicting with anchor model
    Y_pred_test_anchor = anchor.predict(X_test_scaled)
    ```
  - Hilbert schmidt independence criteria (HSIC) regularization [paper](https://proceedings.mlr.press/v162/saengkyongam22a/saengkyongam22a.pdf)


## Getting Started

### Google Colab (recommended for the challenge)

1. Open and run getting_started.ipynb in colab

### Local conda setup (optional)

1. Install conda & mamba.
2. Create conda environment:
```
mamba env create -f environment.yml
```
3. Run `gettiong_started.ipynb`.

### Data size note

- The `lite` data download is ~500 MB and is generally feasible on Colab.
- Expect a few minutes for download/extract and prefer keeping work inside one active session.
- If bandwidth is limited, start with exploratory cells first and run short training (`max_epochs`) during the challenge.