# PhysioNet Challenge 2026: V19 adaptive full training

This candidate reproduces the successful V18/V17/V14-Large method entirely on
the Challenge server and adapts the short CI training budget to the E1 encoder
that was created during the same run. It contains no model weights trained on
Challenge data and has no pretrained-model fallback.

`train_model` performs the following fixed pipeline:

1. discover every labeled record from the provided `demographics.csv`;
2. canonicalize EEG, EOG, ECG, respiratory, SpO2, and EMG signals into the
   same normalized 30-second representation used by inference;
3. align Human and CAISR annotations;
4. train E1 for 15 FP32 epochs with seed `20260804`, 128 windows per record and
   the original physiology, masked-representation, and device-view objectives;
5. export the EMA-teacher 192D embedding for every eligible 30-second window;
6. for each CI seed, test epochs 1 through 6 using inner validation inside each
   of the three site-wise LOSO folds, choose the earliest near-best epoch in
   each fold, and take the median fold epoch;
7. compare those adaptive budgets with the V18 budgets `2/4/1` on matched
   three-site OOF predictions; use the adaptive budgets only if Macro AC gains
   at least 0.002, at least two sites improve, and Worst/site-drop guards pass;
8. train the local five-minute and full-night CI models for seeds `20260806`,
   `20260807`, and `20260808` on all records using the gated budgets;
9. refit the CreationTime, 90-feature CAISR, and follow-up residual rules from
   the provided training labels; and
10. save only the newly trained encoder, sequence ensemble, residual rules, and
   a complete training audit in the official model folder.

The probability-to-binary threshold remains 0.5. V18 already led the Reward
leaderboard at this operating point, so V19 does not add a last-day threshold
search that could overfit the three public training sites.

Inference remains record-wise. A valid current-record `CreationTime` has
priority; otherwise the current EDF header is used, followed by a zero
adjustment. Missing or invalid CAISR also yields a zero CAISR adjustment. No
hidden-cohort statistics are read or estimated.

If adaptive selection fails its engineering checks, the pipeline fails closed
to the scored V18 `2/4/1` budgets. The production protocol is otherwise locked.
Reduced records or epochs are available
only when `PSG4CI_V18_TEST_MODE=1` is explicitly set for engineering tests.
