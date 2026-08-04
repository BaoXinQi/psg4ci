# PhysioNet Challenge 2026: protected full-H residual ensemble

This repository is a George B. Moody PhysioNet Challenge 2026 entry for predicting cognitive impairment from overnight polysomnography. It follows the official Python entry interface: `train_model.py`, `run_model.py`, and `helper_code.py` are unchanged from the official template, while the implementation is in `team_code.py` and supporting modules.

## Model

The prediction is the mean logit from three seeded models. Each model contains:

- a structured anchor MLP using age, BMI, sex, race, ethnicity, and CAISR-derived sleep-stage/event summaries;
- modality-group PCA projections of full-night handcrafted PSG features;
- a gated residual head whose correction is added to the anchor logit.

The PSG residual branch and all preprocessing objects are packaged as pretrained artifacts. During the official training stage, the code processes every labeled record in the supplied training set to recreate demographics and CAISR features, then performs one epoch of low-learning-rate supervised updating of each anchor MLP's final linear layer. All other parameters remain frozen. This makes training inexpensive while ensuring that the submitted code continues training on the supplied data.

At inference, the code reads the raw physiological EDF and CAISR annotation EDF, constructs the required canonical channels, extracts the full-night features, and averages the three model logits. If the physiological branch fails for an individual record, the entry falls back to its demographics and CAISR anchor instead of failing the complete run.

## Pretraining data

The packaged model was trained only on the official PhysioNet Challenge 2026 Large training set:

- 6,600 labeled overnight records;
- 498 positive cognitive-impairment labels;
- training sites S0001, I0002, and I0006;
- official demographics, raw PSG, CAISR annotations, and cognitive labels.

No validation/test labels, supplementary labels, external datasets, or test-time target adaptation were used. The final packaged artifacts were fitted on all 6,600 official Large training records with seeds 20262259, 20262260, and 20262261.

Local diagnostics for the frozen architecture were an official challenge score of 0.815681 under the fixed five-fold split and 0.590174 under leave-one-site-out evaluation. These values describe local validation only and are not claimed leaderboard scores.

## Usage

Install dependencies in Python 3.10, or build the included Docker image. PyTorch 2.0.1 CPU is installed separately in the Dockerfile.

```bash
python train_model.py -d /path/to/training_data -m /path/to/model -v
python run_model.py -d /path/to/holdout_data -m /path/to/model -o /path/to/outputs -v
```

Build and run the container with the same commands described by the official example repository. The entry does not require a GPU and does not access the network at training or inference time.

## Files

- `team_code.py`: official training, loading, and inference entry points.
- `online_features.py`: raw EDF to CAISR and PSG feature bridge.
- `pretrained_model/`: the three full-data pretrained models and fitted preprocessors.
- `caisr_feature_extractor.py`, `psg_feature_extractor.py`, `preprocessing_primitives.py`, `channel_mapper.py`: deterministic feature extraction.
- `train_large_structured_baselines_v1.py`: model class definitions required to load the packaged artifacts.

The submission format is based on the official `physionetchallenges/python-example-2026` repository at commit `c0bcd78ddb892290b9d218be9be19660f4d8bdf3`.
