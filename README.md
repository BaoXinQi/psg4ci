# PhysioNet Challenge 2026: frozen E1 Raw-only full-night model

This entry uses a frozen multimodal 30-second PSG encoder followed by a hierarchical full-night Transformer. The packaged model was pretrained and fitted only on the official 6,600-record Large training set.

The inference path is:

1. canonicalize and robust-normalize EEG, EOG, ECG, respiratory, SpO2, and EMG channels from the raw EDF;
2. encode eligible 30-second windows into frozen 192-dimensional E1 embeddings;
3. aggregate ten windows into each five-minute token with a two-layer local Transformer;
4. aggregate the full night with a three-layer Transformer and predict cognitive impairment from its CLS representation.

The final sequence model was fitted on all 6,600 training records for four fixed epochs with seed `20260806`. Demographics, CAISR annotations, and handcrafted PSG features are not final model inputs.

During the official training stage, all pretrained artifacts are copied into the model directory, a small deterministic set of labeled records is processed as an end-to-end training audit, and only the final scalar bias receives a negligible data-dependent update. The E1 encoder and the rest of the sequence model remain frozen.
