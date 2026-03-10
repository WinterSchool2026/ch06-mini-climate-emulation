# Mini Climate Emulation Challenge

<a target="_blank" href="https://colab.research.google.com/github/WinterSchool2026/ch06-mini-climate-emulation/blob/ch06/getting_started.ipynb">
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

1. How should we handle non-normal or long-tailed distributions in the input variables?
2. What is the optimal train/ validation split? Should historical data be included in training?
3. Which model class gives the best accuracy–compute trade-off for this task?
4. Should we predict the evaluation indices directly, or predict full climate fields and compute indices afterward?
5. How can we scale the same training pipeline from the lite dataset to the full dataset?
6. Which variables and regions contribute most to model error, and can targeted weighting help?
7. How sensitive are results to lead time (near-term vs long-term horizons) across SSP scenarios?
8. Do we need variable-specific loss weighting so low-variance targets are not overshadowed?
9. How stable are results across random seeds and initialization choices?
10. Which calibration method best improves probabilistic reliability of predictions?

## Recommended reading material

Related but not required readings & resources:
  - [NorESM2 Models](https://gmd.copernicus.org/articles/13/6165/2020/)
  - [Emulation challenge example: ClimateBench](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021MS002954)
  - [Another emulation challenge example](https://ui.adsabs.harvard.edu/abs/2023AGUFMGC33F1206L/abstract)
  - [Some nice code for climate emulation](https://github.com/blutjens/climate-emulator-tutorial)
  - [Indices used for evaluation](https://climpact-sci.org/indices/)


## Getting Started

### Google Colab (recommended for the challenge)

1. Create a new Colab notebook and switch runtime to GPU (optional but recommended).
2. Clone this repository and move into it:
```
!git clone https://github.com/IPL-UV/ClimX.git
%cd ClimX
```
3. Install dependencies:
```
!pip install -r requirements-colab.txt
```
4. Open and run `playground.ipynb` (Colab-first starter) or execute:
```
from pathlib import Path
from src.utils.hugging_face_utils import get_dataset_from_hf

data_path = Path('data/')
DATA_VERSION = 'lite'

get_dataset_from_hf(data_path, variant=DATA_VERSION)
```
5. Start training baselines and iterating on your chosen track.

### Local conda setup (optional)

1. Install conda & mamba.
2. Create conda environment:
```
mamba env create -f environment.yml
```
3. Run `playground.ipynb`.

### Data size note

- The `lite` data download is ~500 MB and is generally feasible on Colab.
- Expect a few minutes for download/extract and prefer keeping work inside one active session.
- If bandwidth is limited, start with exploratory cells first and run short training (`max_epochs`) during the challenge.