# PhysioNet Challenge 2026: Domain-robust Raw ensemble with site-relative date-tail risk

This entry uses a frozen multimodal 30-second PSG encoder followed by a hierarchical full-night Transformer. The packaged model was pretrained and fitted only on the official 6,600-record Large training set.

The inference path is:

1. canonicalize and robust-normalize EEG, EOG, ECG, respiratory, SpO2, and EMG channels from the raw EDF;
2. encode eligible 30-second windows into frozen 192-dimensional E1 embeddings;
3. aggregate ten windows into each five-minute token with a two-layer local Transformer;
4. aggregate the full night with a three-layer Transformer and average three CI-model logits trained with a weak training-only site adversary;
5. rank CreationTime within each inference site and add a nonnegative recent-tail risk fitted from legal within-site, age-matched training pairs.

The three sequence models were fitted on all 6,600 training records with seeds `20260806`, `20260807`, and `20260808`. A weak gradient-reversal site head was used only during training and is absent from inference. CreationTime is the only demographic field used by the final predictor. CAISR annotations and handcrafted PSG features are not final model inputs.

During the official training stage, the tail-risk coefficient is refitted from all available labeled records by minimizing the site-macro legal-pair logistic loss. At inference, only the unlabeled CreationTime distribution of each site is used to compute each record's percentile. The packaged Raw artifacts are copied into the model directory, a small deterministic set of records is processed as an end-to-end training audit, and only the final scalar biases receive a negligible data-dependent update. The E1 encoder and the rest of the sequence models remain frozen.
