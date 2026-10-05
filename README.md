# Matcha: Hierarchical Multimodal Sleep Representation Learning

Code for **Hierarchical Multimodal Sleep Representation Learning for Cross-Site
Cognitive Impairment Prediction**, Computing in Cardiology 2026.

**Recommended version: V18, the fully server-trained final entry.** `main`
restores its training and inference code; the new changes are documentation
only. Tag [v18-final](https://github.com/BaoXinQi/psg4ci/tree/v18-final) identifies
the exact submitted commit `7f2eae2fe4f170c87a21943b24ea4e217eabf67f`
(submission **2692**). V20 remains available on
[archive/v20-e1-window-domain-adversarial](https://github.com/BaoXinQi/psg4ci/tree/archive/v20-e1-window-domain-adversarial),
and all original history and existing branches are retained.

Matcha ranked **1st during official-phase validation**, with best scores of
**0.847 on Small** and **0.844 on Large**, and **2nd among 42 eligible teams on
the final test set**, with AC-AUROC **0.727**. These are different evaluation
settings, not interchangeable estimates of generalization.

## Final V18 pipeline

V18 contains no packaged Challenge-trained weights and no pretrained-model
fallback. It trains the complete pipeline on the supplied training data:

1. Discover labeled records from `demographics.csv` and map raw EDF files.
2. Canonicalize EEG, EOG, ECG, respiration, SpO2 and EMG into normalized
   30-second windows, retaining channel and signal-validity masks.
3. Align available Human and CAISR stage/event labels; mask missing targets.
4. Train E1 for **15 FP32 epochs**, seed **20260804**, **128 windows per record
   per epoch**, with sleep supervision, latent prediction and acquisition-style
   consistency objectives.
5. Export the EMA teacher's **192-D** embeddings for every eligible window.
6. Train a **two-layer local Transformer** over ten windows per five-minute
   token and a **three-layer full-night Transformer**. Sequence seeds
   **20260806 / 20260807 / 20260808** use fixed **2 / 4 / 1** epochs and
   equal-weight logit averaging. Training uses weighted BCE, **0.15** same-site
   age-matched pairwise loss and a weak training-only sequence site adversary;
   the adversarial head is not deployed.
7. Refit recording-date, 90-feature CAISR-summary and follow-up-eligibility
   residuals from the supplied training labels. Add **0.375** times their sum
   to the PSG ensemble logit.
8. Save the newly trained encoder, sequence ensemble, residuals, artifact
   hashes and training audit in the requested model directory.

### Record-wise inference

Predictions use the current record and training-fitted parameters, not
statistics of other target records. A valid current-record `CreationTime`
takes priority; otherwise its EDF header is tried, followed by zero
date-dependent adjustment. Missing/invalid CAISR yields zero CAISR adjustment.
Short or unreadable PSGs have finite fallback predictions. V18 performs no
target-cohort normalization or cohort-level reranking.

### Reproduction

Obtain the data from the [official data page](https://moody-challenge.physionet.org/2026/#data).
Production training requires a CUDA GPU and substantial temporary disk space.
Use a **dedicated scratch directory**: training clears its own previous
workspace. Never set `PSG4CI_V18_WORKSPACE` to your dataset or an unrelated
directory.

Build the pinned environment and train; replace the host paths below:

```bash
docker build -t matcha-v18 .
docker run --rm --gpus all --shm-size=8g \
  -v /path/to/training_set_large:/data:ro \
  -v /path/to/model:/model \
  -v /path/to/dedicated_scratch:/scratch \
  -e PSG4CI_V18_WORKSPACE=/scratch/psg4ci_v18_full_training \
  matcha-v18 python train_model.py -d /data -m /model -v
```

Run inference:

```bash
docker run --rm --gpus all \
  -v /path/to/holdout_data:/data:ro \
  -v /path/to/model:/model:ro \
  -v /path/to/predictions:/outputs \
  matcha-v18 python run_model.py -d /data -m /model -o /outputs -v
```

Reduced records/epochs require explicitly enabling
`PSG4CI_V18_TEST_MODE=1`; this is an engineering test, not the scored protocol.
Fixed seeds do not guarantee bitwise-identical GPU training across environments.
Hidden validation/test labels are not distributed here, so their scores cannot
be recomputed from the public training set alone.

## Official results and version history

All scores below are **age-conditioned AUROC**. Official-phase I0004 validation
included recording dates. For final evaluation, organizers removed sleep-study
dates and evaluated the selected V18 on date-free I0004 and independent I0007.

| Version | Main change | I0004 AC-AUROC | Reward | Status / source |
| --- | --- | ---: | ---: | --- |
| V1 | Structured demographics + CAISR + handcrafted physiology | 0.534 | - | Early baseline; [0666e21](https://github.com/BaoXinQi/psg4ci/tree/0666e21) |
| V2 | Frozen E1 + single-seed hierarchical Raw model | No score | - | Short-PSG inference failure; ID 2435, [7125d21](https://github.com/BaoXinQi/psg4ci/tree/7125d21) |
| V3 | Same-site ranking, three seeds, robust inference | 0.721 | 0.122 | ID 2464; [6ac7df5](https://github.com/BaoXinQi/psg4ci/tree/6ac7df5) |
| V4 | Weak sequence-domain-adversarial Raw family | Not submitted separately | - | [v4-domain-robust](https://github.com/BaoXinQi/psg4ci/tree/v4-domain-robust) |
| V5 | Absolute current-record date residual | Not submitted separately | - | Record-wise backup; [fallback branch](https://github.com/BaoXinQi/psg4ci/tree/fallback/v5-date-residual-ci-fixed) |
| V6-V10 | Site-relative date-tail/follow-up-window variants | Not submitted separately | - | Cohort-dependent exploration; branches below |
| V11 | Domain-Raw + five-year eligibility gap | No score | - | Short-PSG failure, ID 2491; not selected; [aa11f4a](https://github.com/BaoXinQi/psg4ci/tree/aa11f4a) |
| V13 | Record-wise Domain-Raw + date + CAISR residual | No score | - | Short-PSG failure, ID 2502; [392f610](https://github.com/BaoXinQi/psg4ci/tree/392f610) |
| V14 Large | Record-wise date + CAISR + training-derived follow-up, weight 0.5 | 0.844 | 0.291 | ID 2503; [8bf2f93](https://github.com/BaoXinQi/psg4ci/tree/8bf2f93) |
| V14 Small | Same code, residual fitting on Small | 0.847 | 0.286 | ID 2541; same [8bf2f93](https://github.com/BaoXinQi/psg4ci/tree/8bf2f93) |
| V15 | Add SessionID, Age, BMI and Sex residual | 0.828 | 0.279 | ID 2551; excluded; [6ea3d78](https://github.com/BaoXinQi/psg4ci/tree/6ea3d78) |
| V16 | V14 + current-record EDF date fallback | 0.844 | 0.291 | ID 2580; [7bd47bc](https://github.com/BaoXinQi/psg4ci/tree/7bd47bc) |
| V17 | Residual weight 0.5 to 0.375 | 0.844 | 0.319 | ID 2604; [909eb62](https://github.com/BaoXinQi/psg4ci/tree/909eb62) |
| **V18** | **Full server retraining, fixed sequence epochs** | **0.840** | **0.336** | **Selected ID 2692; [v18-final](https://github.com/BaoXinQi/psg4ci/tree/v18-final)** |
| V19 | Adaptive sequence-epoch selection | 0.839 | 0.263 | ID 2786; excluded; [b369516](https://github.com/BaoXinQi/psg4ci/tree/b369516) |
| V20 | V19 + weak E1 window-level site adversary | 0.821 | 0.344 | ID 2824; excluded; [archive](https://github.com/BaoXinQi/psg4ci/tree/archive/v20-e1-window-domain-adversarial) |

**Selected V18 final evaluation:**

| Evaluation | AC-AUROC | AUROC | AUPRC | Reward |
| --- | ---: | ---: | ---: | ---: |
| Date-free validation, I0004 | 0.737 | 0.805 | 0.280 | 0.122 |
| Final test, I0007 | 0.727 | 0.773 | 0.305 | 0.129 |

Preliminary and final scores differ in date availability and target population,
not just code version. Legacy frozen-model entries remain research archives,
not substitutes for the final fully server-trained protocol.

### Exploratory branches

- [V6](https://github.com/BaoXinQi/psg4ci/tree/candidate/v6-tail): site-relative CreationTime tail.
- [V7](https://github.com/BaoXinQi/psg4ci/tree/candidate/v7-domain-tail): tail rule + Domain-Raw.
- [V8](https://github.com/BaoXinQi/psg4ci/tree/candidate/v8-horizon2192): 2,192-day follow-up window.
- [V9](https://github.com/BaoXinQi/psg4ci/tree/candidate/v9-domain-horizon2192): window rule + Domain-Raw.
- [V10](https://github.com/BaoXinQi/psg4ci/tree/candidate/v10-eligibility-gap5y): five-year eligibility gap.
- [V11](https://github.com/BaoXinQi/psg4ci/tree/candidate/v11-domain-eligibility-gap5y): gap + Domain-Raw.

These branches derive a date reference from the target cohort. Their local
results are **transductive diagnostics**, not independent record-wise screening.
V11 reached local Macro **0.8199** / Worst **0.7174**, but obtained no official
hidden score. This approach was excluded after organizer clarification.
Keeping the code available does not imply approval for Challenge deployment.

## Local development findings

**Macro** is the unweighted mean of the three held-out-site AC-AUROCs; **Worst**
is their minimum. These are **CI-label LOSO** results: the held-out CI labels
were excluded, but E1 auxiliary pretraining had seen PSG from all three
training sites. They are not strict encoder-domain LOSO.

| Controlled comparison | Local result | Decision |
| --- | --- | --- |
| Structured baseline | Random five-fold AC 0.8157 vs LOSO about 0.5902 | Mixed-site CV was optimistic |
| Frozen E1 pooled statistics | Macro 0.6489 | Embeddings contain patient-level CI information |
| Hierarchical Raw, one seed | Macro 0.7148 | Better than pooling |
| M1: AC-only selection, cross-site pairs, three seeds | Macro 0.7110 | Objective comparison, not final model |
| M2: same-site pairs, three seeds | Macro 0.7273 / Worst 0.6688 | Strong Raw-only baseline |
| M3 sequence SSL | Macro 0.7147 | No material gain; closed |
| Fusion adaptation (local experiment also called "v4") | Single-seed Macro 0.7007 vs M2 0.7181 | Closed; distinct from submission V4 Domain-Raw |
| E1.2 | Single-seed Macro 0.6812; fixed E1/E1.2 blend 0.7110 vs E1 0.7181 | Both failed; closed |
| V14 residual stack | Macro 0.8039 / Worst 0.7128 | Retained; chronology supplied most of the gain |
| V15 demographic residual | Macro 0.8119 / Worst 0.7287; official 0.844 to 0.828 | Local gain did not transfer; excluded |
| Independent E1 seed | Single-seed Raw Macro 0.7059 vs 0.7181 | Fixed-family blend also failed |
| Cross-modal encoder, three sequence seeds + V14 residual | Macro 0.8025 / Worst 0.7133 vs V14 0.8039 / 0.7128 | Gate failed |
| Fixed 50/50 E1/cross-modal blend + V14 residual | Macro 0.8033 / Worst 0.7134 | Gate failed |
| Shared-private representations | Single-seed Macro 0.6899 vs 0.7181 | Closed |
| Intra-window token moments | Single-seed Macro 0.6954 vs 0.7181 | Closed |
| CAISR temporal FiLM | Macro 0.7153 vs matched Domain-Raw 0.7257 | Closed |
| Absolute SpO2 trajectory branch | Single-seed Macro 0.7043 vs 0.7181 | Closed |
| Residual-weight sensitivity | Broad near-flat interval around 0.375-0.685; site-specific optima differed | No precise optimum claimed; 0.375 was a coarse reliance-reduction choice |

Some local audit scripts are not packaged as standalone releases in the
submission repository. These summaries explain decisions; they do not imply
every diagnostic can be reproduced from an existing branch alone. Negative
results apply to the tested protocols, not every implementation of an idea.

## Historical artifacts and attribution

All original commits and existing branches are retained. Early snapshots include
packaged models and `pretrained_model/training_predictions.csv`: identifiers,
labels and predictions for **6,600 deidentified public training records**, not
hidden validation/test labels. V18 does not require this table or historically
packaged weights. Data-derived artifacts are separate from the code license;
check the original data-use terms before redistribution or reuse.

The code retains the **BSD-3-Clause** license and Challenge template attribution.
When citing this work, identify the CinC 2026 paper, exact code tag/commit and
evaluation setting.
