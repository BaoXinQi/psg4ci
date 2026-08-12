# PhysioNet Challenge 2026: V14 plus robust record-wise metadata

This entry uses only the official 6,600-record Large training set. Its inference
path is:

1. canonicalize EEG, EOG, ECG, respiratory, SpO2, and EMG signals;
2. encode 30-second windows with the frozen 192-dimensional E1 encoder;
3. aggregate ten windows into five-minute tokens and then into a whole-night
   representation;
4. average three CI logits trained with a weak training-only site adversary;
5. compute an absolute `CreationTime` residual and a low-capacity residual from
   90 whole-night CAISR summaries;
6. estimate three training-site administrative follow-up cutoffs from the
   training-only `Last_Known_Visit_Date` field and average the corresponding
   six-year follow-up-risk functions; and
7. add half of the combined date, CAISR, and follow-up-risk residual to the Raw
   logit; and
8. add one jointly fitted low-capacity residual from the current record's
   chronologically ordered `SessionID`, Age, observed BMI, and Sex fields.

All inference features are record-wise. The entry does not estimate statistics
from the hidden cohort. Its administrative cutoffs are constants learned from
the training set, not from hidden-set dates. If `CreationTime` is missing, both
date-derived adjustments are zero. If CAISR is missing or unreadable, its
adjustment is zero. A CAISR file with no valid stage or event support is also
treated as unavailable instead of being imputed into a nonzero residual.
The four fields are handled independently. A missing or unparsable field has
exactly zero contribution; there is no explicit missingness feature and no
hidden-cohort imputation or normalization. No other record is inspected.

The packaged Raw and CAISR artifacts were fitted on the official Large training
set. During the official training stage, the absolute-date and follow-up-risk
rules are refitted from the available training labels and training-only
last-visit dates. Six records are processed as an end-to-end Raw audit, and only
the final sequence biases receive a negligible data-dependent update. The E1
encoder and sequence backbones remain frozen.
