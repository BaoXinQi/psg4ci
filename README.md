# PhysioNet Challenge 2026: Domain-Raw + record-wise date + CAISR

This entry uses only the official 6,600-record Large training set. Its inference
path is:

1. canonicalize EEG, EOG, ECG, respiratory, SpO2, and EMG signals;
2. encode 30-second windows with the frozen 192-dimensional E1 encoder;
3. aggregate ten windows into five-minute tokens and then into a whole-night
   representation;
4. average three CI logits trained with a weak training-only site adversary;
5. add an absolute `CreationTime` residual standardized with constants learned
   from the training set; and
6. add a low-capacity residual from 90 whole-night CAISR stage, confidence,
   arousal, respiratory, limb, duration, and validity summaries.

All inference features are record-wise. The entry does not estimate statistics
from the hidden cohort. If `CreationTime` is missing, its adjustment is zero. If
the CAISR annotation file is missing or unreadable, its adjustment is zero, so
the predictor falls back to the domain-Raw model plus any available date input.

The packaged Raw and CAISR artifacts were fitted on the official Large training
set. During the official training stage, the absolute date coefficient is
refitted from available labeled records, the packaged artifacts are copied, six
records are processed as an end-to-end Raw audit, and only the final sequence
biases receive a negligible data-dependent update. The E1 encoder and sequence
backbones remain frozen.
