"""Frozen protocol constants for the official full-training submission."""

from __future__ import annotations


MODEL_SUBDIR = "raw_sequence_v20_e1_domain_adversarial"
E1_SEED = 20260804
E1_EPOCHS = 15
E1_WINDOWS_PER_RECORD = 128
E1_NATURAL_WINDOWS = 64
E1_EVAL_WINDOWS_PER_RECORD = 64
E1_BATCH_SIZE = 16
E1_RECORDS_PER_BATCH = 1
E1_LEARNING_RATE = 6e-4
E1_WEIGHT_DECAY = 1e-4
E1_EMA_DECAY = 0.996
E1_DOMAIN_REVERSAL_MAX = 0.02
E1_DOMAIN_WARMUP_EPOCHS = 2

SEQUENCE_EPOCHS = {
    20260806: 2,
    20260807: 4,
    20260808: 1,
}
SEQUENCE_SELECTION_MAX_EPOCHS = 6
SEQUENCE_SELECTION_INNER_FRACTION = 0.15
SEQUENCE_SELECTION_NEAR_BEST_AC = 0.002
SEQUENCE_SELECTION_MIN_MACRO_GAIN = 0.002
SEQUENCE_SELECTION_MAX_WORST_DROP = 0.002
SEQUENCE_SELECTION_MAX_SITE_DROP = 0.005
SEQUENCE_BATCH_SIZE = 16
SEQUENCE_LEARNING_RATE = 1e-4
SEQUENCE_WEIGHT_DECAY = 1e-4

RESIDUAL_BLEND_WEIGHT = 0.375
CAISR_L2 = 0.10

# This is an input contract, not a learned artifact. Coefficients and all
# normalization statistics are fitted from the official training set.
CAISR_FEATURE_COLUMNS = (
    "record_duration_hours",
    "record_epoch_count",
    "caisr_file_available",
    "caisr_stage_available",
    "caisr_stage_valid_epoch_fraction",
    "caisr_stage_valid_hours",
    "caisr_stage_n3_fraction_valid",
    "caisr_stage_n3_hours",
    "caisr_stage_n2_fraction_valid",
    "caisr_stage_n2_hours",
    "caisr_stage_n1_fraction_valid",
    "caisr_stage_n1_hours",
    "caisr_stage_rem_fraction_valid",
    "caisr_stage_rem_hours",
    "caisr_stage_wake_fraction_valid",
    "caisr_stage_wake_hours",
    "caisr_stage_sleep_fraction_valid",
    "caisr_stage_distribution_entropy",
    "caisr_stage_transition_fraction",
    "caisr_stage_transitions_per_valid_hour",
    "caisr_sleep_onset_min",
    "caisr_sleep_onset_fraction_recording",
    "caisr_waso_fraction_valid_after_onset",
    "caisr_wake_bouts_after_onset",
    "caisr_longest_wake_bout_after_onset_min",
    "caisr_rem_latency_from_sleep_onset_min",
    "caisr_stage_run_count",
    "caisr_stage_runs_per_valid_hour",
    "caisr_stage_probability_available",
    "caisr_stage_probability_valid_epoch_fraction",
    "caisr_prob_n3_mean",
    "caisr_prob_n3_std",
    "caisr_prob_n3_p10",
    "caisr_prob_n3_p90",
    "caisr_prob_n2_mean",
    "caisr_prob_n2_std",
    "caisr_prob_n2_p10",
    "caisr_prob_n2_p90",
    "caisr_prob_n1_mean",
    "caisr_prob_n1_std",
    "caisr_prob_n1_p10",
    "caisr_prob_n1_p90",
    "caisr_prob_rem_mean",
    "caisr_prob_rem_std",
    "caisr_prob_rem_p10",
    "caisr_prob_rem_p90",
    "caisr_prob_wake_mean",
    "caisr_prob_wake_std",
    "caisr_prob_wake_p10",
    "caisr_prob_wake_p90",
    "caisr_prob_max_mean",
    "caisr_prob_max_std",
    "caisr_prob_max_p10",
    "caisr_prob_entropy_mean",
    "caisr_prob_entropy_std",
    "caisr_prob_low_confidence_fraction_lt_0_5",
    "caisr_prob_low_confidence_fraction_lt_0_6",
    "caisr_prob_argmax_stage_agreement",
    "caisr_stage_probability_source_count",
    "caisr_arousal_available",
    "caisr_arousal_mean_valid_ratio",
    "caisr_arousal_valid_epoch_fraction",
    "caisr_arousal_positive_epoch_fraction_all",
    "caisr_arousal_positive_epoch_fraction_valid",
    "caisr_arousal_event_onsets",
    "caisr_arousal_event_onsets_per_hour",
    "caisr_arousal_longest_positive_run_min",
    "caisr_arousal_mean_positive_ratio",
    "caisr_arousal_p90_positive_ratio",
    "caisr_respiratory_available",
    "caisr_respiratory_mean_valid_ratio",
    "caisr_respiratory_valid_epoch_fraction",
    "caisr_respiratory_positive_epoch_fraction_all",
    "caisr_respiratory_positive_epoch_fraction_valid",
    "caisr_respiratory_event_onsets",
    "caisr_respiratory_event_onsets_per_hour",
    "caisr_respiratory_longest_positive_run_min",
    "caisr_respiratory_mean_positive_ratio",
    "caisr_respiratory_p90_positive_ratio",
    "caisr_limb_available",
    "caisr_limb_mean_valid_ratio",
    "caisr_limb_valid_epoch_fraction",
    "caisr_limb_positive_epoch_fraction_all",
    "caisr_limb_positive_epoch_fraction_valid",
    "caisr_limb_event_onsets",
    "caisr_limb_event_onsets_per_hour",
    "caisr_limb_longest_positive_run_min",
    "caisr_limb_mean_positive_ratio",
    "caisr_limb_p90_positive_ratio",
    "caisr_annotation_dataset_count",
)
