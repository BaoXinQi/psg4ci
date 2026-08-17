# PhysioNet Challenge 2026: V18 full training

This candidate reproduces the successful V17/V14-Large method entirely on the
Challenge server. It contains no model weights trained on Challenge data and
has no pretrained-model fallback.

`train_model` performs the following fixed pipeline:

1. discover every labeled record from the provided `demographics.csv`;
2. canonicalize EEG, EOG, ECG, respiratory, SpO2, and EMG signals into the
   same normalized 30-second representation used by inference;
3. align Human and CAISR annotations;
4. train E1 for 15 FP32 epochs with seed `20260804`, 128 windows per record and
   the original physiology, masked-representation, and device-view objectives;
5. export the EMA-teacher 192D embedding for every eligible 30-second window;
6. train the fixed local five-minute and full-night CI model for seeds
   `20260806`, `20260807`, and `20260808` for 2, 4, and 1 epochs;
7. refit the CreationTime, 90-feature CAISR, and follow-up residual rules from
   the provided training labels; and
8. save only the newly trained encoder, sequence ensemble, residual rules, and
   a complete training audit in the official model folder.

Inference remains record-wise. A valid current-record `CreationTime` has
priority; otherwise the current EDF header is used, followed by a zero
adjustment. Missing or invalid CAISR also yields a zero CAISR adjustment. No
hidden-cohort statistics are read or estimated.

The production protocol is locked. Reduced records or epochs are available
only when `PSG4CI_V18_TEST_MODE=1` is explicitly set for engineering tests.
