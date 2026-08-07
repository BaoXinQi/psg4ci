# PhysioNet Challenge 2026: frozen E1 Raw full-night ensemble

This entry uses a frozen multimodal 30-second PSG encoder followed by a hierarchical full-night Transformer. The packaged model was pretrained and fitted only on the official 6,600-record Large training set.

The inference path is:

1. canonicalize and robust-normalize EEG, EOG, ECG, respiratory, SpO2, and EMG channels from the raw EDF;
2. encode eligible 30-second windows into frozen 192-dimensional E1 embeddings;
3. aggregate ten windows into each five-minute token with a two-layer local Transformer;
4. aggregate the full night with a three-layer Transformer and predict cognitive impairment from its CLS representation.

The final predictor is an equal-weight logit ensemble of three sequence models fitted on all 6,600 training records. The locked members use seeds `20260806`, `20260807`, and `20260808` for 2, 4, and 1 epochs, respectively. Each member uses the same-site age-pair objective selected by site-wise LOSO. Demographics, CAISR annotations, and handcrafted PSG features are not final model inputs.

During the official training stage, all pretrained artifacts are copied into the model directory, a small deterministic set of labeled records is processed as an end-to-end training audit, and only each member's final scalar bias receives the same negligible data-dependent update. The E1 encoder and the rest of all three sequence models remain frozen.
