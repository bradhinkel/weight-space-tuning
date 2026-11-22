PRIOR WORK (read-only). Everything here was produced before this session, without the test split.

dfr/  The current best known repair: Deep Feature Reweighting (DFR), a published method that refits only the last
      layer. It was run with 5 random seeds (recipe in dfr/recipe.md). dfr/seed<k>/fc.weight.npy and fc.bias.npy
      are drop-in replacements for the model's fc.weight [2, 2048] and fc.bias [2]: save_edit them to run DFR.
      dfr/practice_scores.json gives each seed's practice scores. They are IN-SAMPLE: DFR was fitted on these
      same practice images, so they overstate how DFR does on new images. Judge DFR on practice images it did not
      fit (for example by re-running the recipe on part of the practice set) before comparing it with anything.

runs/ 18 earlier sessions in which Fable 5.2 repaired this same model by editing its weights, under this task's
      rules. runs/index.csv lists them. 9 had fitting refused ("refused": weights only from activation statistics
      and reasoning) and 9 had fitting allowed ("allowed"); budgets were 10, 25 or 50 forward passes, 3 runs each.
      Each run folder has its submitted tensors (submitted/*.npy, the same names and shapes as the model; tensors
      not present were left as the original), its final answer (report.md) and its working notes (notes.md).
      Every practice number in them is a practice-set measurement, often in-sample for that run's own edit.
