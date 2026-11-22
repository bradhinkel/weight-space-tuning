DFR recipe used (Kirichenko et al. 2022, DFR^Val_Tr):

  - features = input to the last layer (2048-d), from the broken model, eval transform;
  - StandardScaler fitted on the reweighting data;
  - l1 logistic regression (sklearn, liblinear);
  - tune C in {1, .7, .3, .1, .07, .03, .01} by worst-group accuracy: fit on a balanced subsample of
    one half of validation, score on the other half;
  - final: 10 fits on group-balanced subsamples of all validation (all of the smallest group, the same
    number drawn from each other group), weights averaged; the scaler is fitted on all validation.
