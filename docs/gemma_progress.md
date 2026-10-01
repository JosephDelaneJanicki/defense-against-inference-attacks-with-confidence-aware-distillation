# Gemma 3 4B Membership-Inference Defense --- Progress Log

**Project:** Defense Against Membership Inference Attacks with
Confidence-Aware Distillation\
**Model track:** Gemma 3 4B PT\
**Base model:** `google/gemma-3-4b-pt`\
**Hardware:** NVIDIA GeForce RTX 4090 (23.51 GB VRAM)\
**Primary precision:** BF16, no quantization\
**Status:** Undefended target and shadow-MIA experiments complete;
contextual-control experiment complete; defense pipeline in development.

------------------------------------------------------------------------

## 1. Purpose of the Gemma Experiment

The earlier project work studied membership-inference attacks (MIAs) on
tabular and image models. The Gemma experiment extends the project to a
causal large language model.

The central question is:

> Can confidence-aware knowledge distillation reduce membership leakage
> from a realistically fine-tuned language model while preserving useful
> language-model behavior?

This requires adapting the experimental design because an autoregressive
language model does not produce one class distribution per example. It
produces a vocabulary distribution at every prediction position in a
sequence.

The Gemma track therefore uses token-level loss/confidence statistics as
attack features and LoRA for computationally practical target, shadow,
and student fine-tuning.

------------------------------------------------------------------------

## 2. Environment and Model Selection

### Hardware/software validation

The environment was verified as:

-   PyTorch: `2.11.0+cu130`
-   Transformers: `5.17.0`
-   CUDA: `13.0`
-   GPU: NVIDIA GeForce RTX 4090
-   VRAM: `23.51 GB`
-   BF16 support: yes

### Why Gemma 3 4B PT?

`google/gemma-3-4b-pt` was selected rather than a substantially larger
model because the 4.3B-parameter model fits comfortably in the RTX 4090
in BF16.

A successful BF16 load showed:

-   Parameters: approximately **4.300B**
-   Model allocation after load: approximately **8.02 GB**
-   Peak during initial forward test: approximately **8.08 GB**
-   Quantization: **none**

This was preferable to beginning with a larger model that would require
quantization or substantial CPU/offload compromises. Keeping the base
model in BF16 makes the experiment cleaner: quantization is not an
additional variable in the privacy results.

Gemma is a gated Hugging Face repository. Initial HTTP 403 errors were
resolved by authenticating the Hugging Face account and obtaining
repository access.

------------------------------------------------------------------------

## 3. Controlled WikiText-103 Dataset

### Dataset

The controlled corpus is:

-   `Salesforce/wikitext`
-   configuration: `wikitext-103-raw-v1`
-   source split: `train`

The tokenizer is Gemma 3's tokenizer.

### Sequence construction

WikiText contains headings, blanks, and short fragments. Nonempty rows
were therefore concatenated into a buffer and tokenized. Every time at
least 256 tokens were available, a 256-token chunk was emitted.

Configuration:

-   Maximum chunk size: **256 tokens**
-   Minimum accepted chunk size: **64 tokens**
-   Constructed sequences: **470,735**
-   Unique sequences after SHA-256 deduplication: **470,723**

The original controlled split used deterministic seed `42`.

### Primary target split

-   Members: **5,000**
-   Nonmembers: **5,000**
-   Validation: **1,000**
-   Utility test: **1,000**

No SHA-256 overlap exists between these splits.

### Membership definition

"Membership" means:

> The sequence was selected for the controlled LoRA fine-tuning of the
> target model.

It does **not** mean that the text was absent from Gemma's original
pretraining corpus. Because Gemma's original training data are not
controlled by this experiment, the experiment measures leakage induced
by the controlled fine-tuning stage.

This distinction became especially important when evaluating possible
WikiText/context confounds later in the experiment.

------------------------------------------------------------------------

## 4. Untouched Gemma Negative Control

Before fine-tuning, the untouched base Gemma model was scored on the
controlled member/nonmember population.

### Results

  Feature                ROC-AUC
  ------------------- ----------
  -NLL                  0.512090
  -Perplexity           0.512090
  Median token loss     0.505528
  Max token loss        0.498016
  Std token loss        0.516189

### Interpretation

The untouched model is close to chance on the controlled membership
assignment.

This is an essential negative control. The labels were assigned after
construction of the WikiText sequence population, so an untouched model
should not know which examples the experiment later designated as LoRA
members.

The near-0.50 result supports the interpretation that later increases in
attack performance are associated with controlled fine-tuning rather
than an obvious pre-existing split artifact.

------------------------------------------------------------------------

## 5. Realistic LoRA Target

The primary target was fine-tuned using LoRA rather than full-parameter
fine-tuning.

### LoRA configuration

-   Rank: **8**
-   Alpha: **16**
-   Dropout: **0.05**
-   Target modules:
    -   `q_proj`
    -   `k_proj`
    -   `v_proj`
    -   `o_proj`
-   Trainable parameters: **5,949,440**
-   Total parameters: **4,306,028,912**
-   Trainable fraction: **0.1382%**

### Training configuration

-   Members: **5,000**
-   Validation: **1,000**
-   Epochs: **2**
-   Learning rate: **2e-4**
-   Per-device batch size: **1**
-   Gradient accumulation: **8**
-   Effective batch size: **8**
-   BF16
-   No quantization
-   Gradient checkpointing enabled
-   Optimizer: AdamW

### Seed-42 target result

Initial validation loss:

**2.581326**

Final validation loss:

**2.245347**

Training loss:

**\~2.203**

Training time:

**\~1,747 seconds (\~29.1 minutes)**

Peak allocated GPU memory:

**\~8.91 GB**

### Why LoRA?

LoRA allows repeated target, shadow, and defense experiments on the RTX
4090 without introducing model quantization. It also provides a
realistic parameter-efficient fine-tuning setting for modern LLM
adaptation.

------------------------------------------------------------------------

## 6. Direct Token-Statistic Membership Attacks

A causal LM emits one vocabulary distribution for every next-token
prediction. The attack representation therefore uses sequence-level
summaries of token losses rather than a single classification confidence
vector.

Features used:

1.  NLL
2.  Median token loss
3.  Minimum token loss
4.  Maximum token loss
5.  Standard deviation of token loss
6.  Min-K 10%
7.  Min-K 20%
8.  Min-K 30%
9.  Min-K 40%

### Seed-42 realistic LoRA results

  Feature                    ROC-AUC
  ------------------- --------------
  -NLL                      0.532789
  -Perplexity               0.532789
  Median token loss         0.521164
  Max token loss            0.528975
  Std token loss        **0.547462**
  Min-K 10%             **0.545011**
  Min-K 20%                 0.542800
  Min-K 30%                 0.540092
  Min-K 40%                 0.538160

NLL statistics:

-   Members mean NLL: **2.177798**
-   Nonmembers mean NLL: **2.229231**

### Min-K interpretation

Min-K focuses on the hardest/worst-predicted fraction of tokens in a
sequence. The intuition is that training may disproportionately improve
probabilities on unusual or difficult tokens that are more diagnostic of
exact memorization.

This feature was not especially meaningful in earlier Adult/CIFAR work,
where outputs have fundamentally different structure. In the
autoregressive Gemma experiment, however, Min-K became one of the
strongest direct membership signals.

------------------------------------------------------------------------

## 7. Learned Attack on Target Scores --- Diagnostic Only

A logistic-regression and random-forest classifier were initially fit by
splitting the target score population itself.

Results:

### Logistic regression

-   Accuracy: **0.532400**
-   F1: **0.525761**
-   ROC-AUC: **0.547012**

### Random forest

-   Accuracy: **0.530400**
-   F1: **0.565024**
-   ROC-AUC: **0.541421**

The strongest RF feature was `std_token_loss`, followed by Min-K
features and maximum token loss.

### Why this was not treated as the final Shokri attack

This classifier trains directly on part of the target population. It is
useful as a diagnostic that a multivariate attack can extract signal,
but it is not the clean shadow-model methodology desired for the final
experiment.

The experiment therefore moved to independent shadow LoRAs.

------------------------------------------------------------------------

## 8. Shokri-Style Shadow LoRA Attack

### Why the classic attack needed adaptation

In a conventional classifier, one example produces one class-probability
vector.

In Gemma:

> one sequence → many next-token vocabulary distributions.

The Shokri idea is retained, but the attack representation is adapted to
the causal-LM output structure. Shadow models are trained on known
member/nonmember populations, their token-level statistics are
extracted, and the attack classifier learns from those shadow
observations.

### Shadow data

Six shadow datasets were created.

Per shadow:

-   Members: **5,000**
-   Nonmembers: **5,000**
-   Validation: **1,000**
-   Total: **11,000**

Across all six:

-   Total unique shadow examples: **66,000**
-   Target overlap: **0**
-   Cross-shadow overlap: **0**

### Shadow training

Each shadow uses the same realistic LoRA configuration as the target.

The complete six-shadow train-and-score pipeline required approximately:

**3.43 hours**

The resulting attack-training population contained:

-   Total observations: **60,000**
-   Members: **30,000**
-   Nonmembers: **30,000**

### Single target-seed Shokri results

Logistic attacker:

-   Accuracy: **0.531900**
-   F1: **0.536489**
-   ROC-AUC: **0.545891**

Random forest:

-   Accuracy: **0.534600**
-   F1: **0.557689**
-   ROC-AUC: **0.544778**

The attack classifiers were fit **only on shadow observations**. Target
labels were used only for final evaluation.

------------------------------------------------------------------------

## 9. Multi-Seed Replication

A single run could reflect favorable or unfavorable random
initialization/training noise. The experiment was therefore repeated
across four target/shadow seeds:

-   **42**
-   **123**
-   **456**
-   **789**

Each seed received its own realistic target LoRA and six-model shadow
attack population.

### Four-seed Shokri attack summary

#### Shadow logistic regression

-   Accuracy: **0.543375 ± 0.009953**
-   Accuracy range: **0.5339--0.5528**
-   Precision: **0.542958 ± 0.010286**
-   Recall: **0.549200 ± 0.005443**
-   F1: **0.546046 ± 0.007593**
-   ROC-AUC: **0.560067 ± 0.015483**
-   ROC-AUC range: **0.546497--0.573502**

#### Shadow random forest

-   Accuracy: **0.543000 ± 0.011846**
-   Accuracy range: **0.5313--0.5536**
-   Precision: **0.539467 ± 0.011230**
-   Recall: **0.588950 ± 0.008945**
-   F1: **0.563100 ± 0.009432**
-   ROC-AUC: **0.557471 ± 0.014146**
-   ROC-AUC range: **0.544731--0.570089**

### Interpretation

The membership signal is modest but reproducible across independently
seeded runs.

The logistic attack averaged approximately **0.560 ROC-AUC**, and the
random forest approximately **0.557 ROC-AUC**.

This is substantially more informative than reporting one favorable run,
and it establishes the multi-seed standard that defended students must
also satisfy.

------------------------------------------------------------------------

## 10. WikiText Context-Confound Concern

A methodological concern was identified before defense evaluation:

> Could the attack be learning that a sequence resembles the
> vocabulary/topic/context of the WikiText material used for fine-tuning
> rather than identifying exact fine-tuning membership?

This is particularly relevant because both members and nonmembers
originate from the same WikiText-103 source, and nearby chunks may share
vocabulary, entities, topics, or prose style.

The untouched Gemma negative control already argued against a gross
split artifact, but it did not completely distinguish exact-example
leakage from local contextual adaptation.

A stronger control was therefore added before evaluating defenses.

------------------------------------------------------------------------

## 11. Adjacent-Context Control

### Important construction limitation discovered

The original dataset builder concatenated nonblank WikiText rows
continuously and did **not** retain article identifiers or article
boundaries.

Therefore, the new control cannot honestly be described as a guaranteed
"same-article" control.

It is instead called an:

> **Adjacent-context control**

This terminology change was made to avoid claiming provenance that the
original preprocessing did not preserve.

### Construction

The exact original 256-token stream was reconstructed:

-   Constructed: **470,735**
-   Unique: **470,723**

Each of the 5,000 exact members was mapped back into that stream using
SHA-256.

For each member, the nearest unique, unused, non-reserved chunk was
selected as its adjacent nonmember.

The original member, nonmember, validation, and utility splits were
excluded from candidate neighbors.

### Match quality

-   Exact members: **5,000**
-   Adjacent nonmembers: **5,000**
-   Random/original nonmembers: **5,000**
-   Immediate-neighbor fraction: **0.9996**
-   Mean absolute offset: **1.0004 chunks**
-   Maximum offset: **2 chunks**

Thus, **99.96%** of controls are exactly one 256-token chunk away from
their matched member.

------------------------------------------------------------------------

## 12. Adjacent-Control Results

Three comparisons were evaluated across all four target seeds:

1.  Exact member vs random nonmember
2.  Exact member vs adjacent nonmember
3.  Adjacent nonmember vs random nonmember

### Selected four-seed mean AUCs

  Feature                Exact vs Random   Exact vs Adjacent   Adjacent vs Random
  -------------------- ----------------- ------------------- --------------------
  NLL                       **0.542393**        **0.532687**             0.510043
  Median token loss             0.528429            0.520944             0.507396
  Max token loss                0.537140            0.530688             0.506847
  Std token loss            **0.560233**        **0.543784**             0.517106
  Min-K 10%                 **0.557095**        **0.544920**             0.512756
  Min-K 20%                 **0.554396**        **0.543359**             0.511618
  Min-K 30%                 **0.551281**        **0.540624**             0.511214
  Min-K 40%                 **0.548971**        **0.538550**             0.510894
  Minimum token loss            0.504253            0.503988             0.500277

### Matched-pair results

For every member, the corresponding immediate adjacent control can be
compared directly.

Mean member-win fractions across four seeds:

  Feature                Member more member-like
  -------------------- -------------------------
  NLL                                 **0.5444**
  Std token loss                     **0.54515**
  Min-K 10%                           **0.5502**
  Min-K 20%                          **0.55705**
  Min-K 30%                           **0.5549**
  Min-K 40%                           **0.5507**
  Minimum token loss                    0.502125

### Interpretation

The overall pattern is:

> **Exact members differ from adjacent nonmembers, while adjacent
> nonmembers are much closer to random nonmembers.**

For example:

-   Min-K 10% exact vs random: **0.5571**
-   Min-K 10% exact vs adjacent: **0.5449**
-   Min-K 10% adjacent vs random: **0.5128**

This supports the interpretation that most of the detected signal is
associated with exact controlled fine-tuning membership rather than
merely being near similar WikiText context.

However, adjacent vs random is not perfectly 0.50. Some small
contextual/distributional effect is detectable, especially for standard
deviation of token loss (**0.5171**). The defensible conclusion is
therefore not that context has zero effect, but that exact membership
retains additional signal after strong local-context matching.

Minimum token loss remains essentially chance, which is also
informative: the easiest/best-predicted token is not useful for
membership, whereas the difficult-token tail captured by Min-K is
substantially more informative.

------------------------------------------------------------------------

## 13. Defense Design

With the undefended leakage replicated and the context confound
investigated, the experiment moved to confidence-aware distillation.

Planned variants:

1.  **Raw-soft**
    -   Ordinary soft-target distillation
    -   Temperature = 1
    -   No confidence clipping
    -   No confidence filtering
    -   Serves as the distillation-only baseline
2.  **Temperature-soft**
    -   Softens the teacher distribution using temperature
    -   Tests whether reducing distribution sharpness reduces membership
        leakage
3.  **Clip-soft**
    -   Caps excessive teacher confidence
    -   Tests whether the highest-confidence teacher outputs
        disproportionately transfer membership information
4.  **Filtered-soft**
    -   Suppresses/removes high-confidence teacher positions according
        to a predetermined confidence rule
    -   Tests whether avoiding the most confident teacher supervision
        improves privacy

The purpose of the raw-soft baseline is critical: if ordinary
distillation already reduces leakage, the confidence-aware variants must
be compared against **raw distillation**, not only against the
undefended teacher.

------------------------------------------------------------------------

## 14. Seed-42 Teacher Top-K Cache

Before student training, the Seed-42 realistic teacher was run over the
controlled 5,000 training-member sequences.

Instead of storing a full 262,208-token probability vector at every
prediction position, the cache stores:

-   Top-64 token IDs
-   Top-64 raw teacher logits
-   Exact teacher top-1 confidence
-   Exact Top-64 captured probability mass
-   Token targets / alignment information
-   Sequence identifiers/hashes

### Cache result

-   Cache size: **0.48 GB**
-   Prediction positions: **1,274,993**
-   K: **64**

### Teacher confidence

-   Mean: **0.526222**
-   P50: **0.468578**
-   P75: **0.827800**
-   P90: **0.993089**
-   P95: **0.999119**
-   P99: **0.999952**
-   Fraction \> 0.85: **0.236560**
-   Fraction \> 0.95: **0.166375**

### Top-64 captured probability mass

-   Mean: **0.930745**
-   P50: **0.982407**
-   P75: **0.998662**
-   P90: **0.999973**
-   P95: **0.999996**
-   P99: **1.000000**

### Why this measurement was done before choosing defense thresholds

Earlier Adult experiments used confidence thresholds such as 0.85 and
0.95. Those values should not simply be copied into the LLM experiment
without checking what they mean for Gemma.

The teacher cache therefore measured Gemma's actual confidence
distribution **before downstream defense/MIA evaluation**.

This avoids selecting thresholds because they happen to produce
favorable attack results.

The measured distribution shows that 0.85 and 0.95 are nontrivial in
Gemma:

-   \~23.7% of token positions exceed 0.85
-   \~16.6% exceed 0.95

These values can now be considered as principled reference points rather
than blindly inherited constants.

------------------------------------------------------------------------

## 15. Why Top-64 Uses a Residual "Other" Bucket

Mean Top-64 captured mass is high (**\~93.1%**) but not 100%.

Simply applying softmax over the cached 64 logits would renormalize them
to sum to 1 and silently pretend the omitted vocabulary had zero teacher
probability.

The raw-soft student instead represents the teacher as:

`{p_1, ..., p_64, p_other}`

where:

`p_other = 1 - Top64_mass`

The student's probabilities are normalized over the full vocabulary.
Probabilities assigned to the teacher's 64 explicit tokens are gathered,
and all remaining student vocabulary probability is collapsed into the
same residual "other" bucket.

This preserves the measured teacher probability mass much more
faithfully while keeping the cache computationally practical.

------------------------------------------------------------------------

## 16. Raw-Soft Student --- Current Stage

A Seed-42 raw-soft student script has been prepared.

### Student initialization

The student begins from a **fresh Gemma 3 4B base model**, not from the
fine-tuned teacher.

It receives a new LoRA adapter with the same architecture:

-   Rank 8
-   Alpha 16
-   Dropout 0.05
-   q/k/v/o projections

### Supervision

The student processes the same input sequences used to obtain teacher
outputs, but it is **not directly optimized with hard next-token
labels**.

Its supervision is the teacher's cached soft distribution:

-   Temperature: **1**
-   No clipping
-   No filtering
-   No hard-label CE

This is why the variant is called **raw-soft** or **unfiltered
distillation**.

### Why raw-soft comes first

Raw-soft answers:

> How much membership leakage is removed simply by knowledge
> distillation?

Only after establishing that baseline can temperature, clipping, and
filtering be credited with improvements beyond ordinary distillation.

### Current status

At the time of this progress log:

-   Seed-42 teacher cache: **complete**
-   Raw-soft student training script: **prepared**
-   Raw-soft Seed-42 student training/evaluation: **next/current run**
-   Temperature-soft: pending
-   Clip-soft: pending
-   Filtered-soft: pending

------------------------------------------------------------------------

## 17. Multi-Seed Defense Plan

The final defense results must use the same four-seed standard as the
undefended experiment:

-   42
-   123
-   456
-   789

Each teacher must generate its **own cache**, and each student must be
distilled from the teacher corresponding to that seed.

Correct structure:

`Teacher(seed S) -> Top-K cache(seed S) -> defense(seed S) -> student(seed S) -> utility + MIA(seed S)`

Seed 42 is being used first as the development/pipeline-validation seed.
This prevents spending several GPU-hours replicating a bug across every
seed and defense variant.

Once Seed 42 raw-soft training, utility scoring, and MIA scoring are
validated, the implementation should be frozen and expanded to all four
seeds.

The eventual comparison matrix is:

  Variant                         42       123       456       789
  -------------------- ------------- --------- --------- ---------
  Undefended teacher               ✓         ✓         ✓         ✓
  Raw-soft               development   pending   pending   pending
  Temperature-soft           pending   pending   pending   pending
  Clip-soft                  pending   pending   pending   pending
  Filtered-soft              pending   pending   pending   pending

Final reporting should use mean ± standard deviation across the four
seeds rather than selecting a favorable run.

------------------------------------------------------------------------

## 18. Experimental Logic So Far

The Gemma experiment has deliberately proceeded in stages:

### Stage A --- Establish a valid controlled membership task

Create deterministic, hash-disjoint member/nonmember populations.

### Stage B --- Negative control

Verify untouched Gemma cannot meaningfully identify the experiment's
later membership assignment.

Result: approximately chance.

### Stage C --- Introduce realistic fine-tuning

Train a small LoRA adapter rather than intentionally forcing extreme
overfitting.

Result: validation loss improves and measurable membership signal
appears.

### Stage D --- Adapt MIA features to causal LMs

Use token-loss statistics and Min-K rather than treating a language
model like a single-label classifier.

Result: difficult-token statistics are informative.

### Stage E --- Proper shadow-model attack

Train attack classifiers exclusively from independently trained shadow
LoRAs.

Result: approximately 0.55 ROC-AUC on the first target.

### Stage F --- Replicate across random seeds

Repeat targets and shadow attacks across four seeds.

Result: logistic **0.5601 ± 0.0155** ROC-AUC and RF **0.5575 ± 0.0141**.

### Stage G --- Challenge the interpretation

Test whether the signal is simply WikiText/context similarity.

Result: exact members remain distinguishable from
almost-immediately-adjacent nonmembers, while adjacent and random
nonmembers are much closer.

### Stage H --- Begin defense

Measure teacher confidence before choosing defense behavior, cache
compact soft targets, and establish raw-soft ordinary distillation as
the first defense baseline.

This ordering is intentional: the project establishes that there is a
reproducible membership signal and investigates an important confound
**before** claiming that a defense reduces it.

------------------------------------------------------------------------

## 19. Key Results Snapshot

### Undefended privacy

-   Untouched Gemma direct NLL AUC: **0.5121**
-   Realistic Seed-42 direct strongest signals: approximately
    **0.545--0.547**
-   Four-seed Shokri logistic AUC: **0.5601 ± 0.0155**
-   Four-seed Shokri RF AUC: **0.5575 ± 0.0141**

### Context control

-   Exact vs random, Min-K 10: **0.5571**
-   Exact vs adjacent, Min-K 10: **0.5449**
-   Adjacent vs random, Min-K 10: **0.5128**
-   Exact vs random, std token loss: **0.5602**
-   Exact vs adjacent, std token loss: **0.5438**
-   Adjacent vs random, std token loss: **0.5171**

### Teacher cache

-   Prediction positions: **1,274,993**
-   Cache size: **0.48 GB**
-   Mean top-1 confidence: **0.5262**
-   Fraction confidence \> 0.85: **23.66%**
-   Fraction confidence \> 0.95: **16.64%**
-   Mean Top-64 probability mass: **93.07%**
-   Median Top-64 probability mass: **98.24%**

------------------------------------------------------------------------

## 20. Immediate Next Steps

1.  Complete Seed-42 raw-soft student training.
2.  Score raw-soft utility on the frozen validation/utility populations.
3.  Score the raw-soft student on the same frozen 5,000-member /
    5,000-nonmember population.
4.  Apply the established MIA methodology without tuning it to the
    defended model.
5.  Compare raw-soft leakage with the Seed-42 undefended teacher.
6.  Validate that the defense pipeline is behaving as intended.
7.  Freeze the pipeline.
8.  Generate teacher caches and raw-soft students for seeds 123, 456,
    and 789.
9.  Add temperature-soft, clipping, and filtering variants using
    thresholds/rules chosen without reference to downstream MIA
    performance.
10. Report privacy and utility jointly as mean ± standard deviation
    across four seeds.

------------------------------------------------------------------------

## 21. Current Working Interpretation

The experiment currently supports the following provisional
interpretation:

1.  Controlled realistic LoRA fine-tuning introduces a small but
    reproducible membership signal into Gemma 3 4B.
2.  The signal is detectable by both direct token statistics and
    independently trained Shokri-style shadow attacks.
3.  Min-K is particularly appropriate for the autoregressive setting
    because difficult-token behavior contains more membership
    information than the easiest token predictions.
4.  Immediate WikiText context explains only part of the observed
    separation; exact fine-tuning members retain additional signal
    against closely adjacent nonmembers.
5.  The remaining research question is whether distillation---and
    specifically confidence-aware manipulation of teacher
    supervision---can suppress this membership signal without
    unacceptable utility loss.

These conclusions remain provisional until the defended multi-seed
experiments are complete.
