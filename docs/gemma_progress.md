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
## Raw-Soft Distillation Defense and Defense-Aware MIA

### Motivation

After establishing the Gemma 3 4B PT membership-inference baseline, the next objective was to test whether knowledge distillation could reduce fine-tuning membership leakage while preserving model utility.

The undefended realistic LoRA target exhibited weak but repeatable membership leakage. Across four target/shadow seeds, the Shokri-style attack produced:

| Attacker | ROC-AUC Mean | ROC-AUC Std | Min | Max |
|---|---:|---:|---:|---:|
| Shadow logistic regression | 0.560067 | 0.015483 | 0.546497 | 0.573502 |
| Shadow random forest | 0.557471 | 0.014146 | 0.544731 | 0.570089 |

This established that leakage was modest but consistently above random chance.

The defense experiments therefore focus on whether distillation can move membership separability toward an AUC of 0.50 without materially degrading held-out language-model performance.

---

## Teacher Top-K Cache

A teacher-output cache was constructed for the Seed-42 realistic LoRA teacher.

Rather than repeatedly evaluating the approximately 4B-parameter teacher during student training, the teacher was evaluated once and its next-token distributions were compressed into a Top-K representation.

Configuration:

- Base model: `google/gemma-3-4b-pt`
- Teacher: Seed-42 realistic LoRA
- Distillation corpus: 5,000 original fine-tuning members
- Sequence length: 256 tokens
- Top-K: 64
- Prediction tokens cached: 1,274,993
- Cache size: approximately 0.48 GB

Teacher confidence statistics:

| Statistic | Confidence |
|---|---:|
| Mean | 0.526222 |
| P50 | 0.468578 |
| P75 | 0.827800 |
| P90 | 0.993089 |
| P95 | 0.999119 |
| P99 | 0.999952 |
| Fraction > 0.85 | 0.236560 |
| Fraction > 0.95 | 0.166375 |

The Top-64 representation retained most of the teacher probability mass:

| Statistic | Top-64 Captured Mass |
|---|---:|
| Mean | 0.930745 |
| P50 | 0.982407 |
| P75 | 0.998662 |
| P90 | 0.999973 |
| P95 | 0.999996 |
| P99 | 1.000000 |

The cache stores raw Top-64 teacher logits together with the exact captured Top-64 probability mass. This allows the student objective to preserve the teacher distribution over the retained tokens while also representing probability mass outside the Top-64 set.

This design avoids storing full-vocabulary logits, which would require substantially more storage.

---

## Adjacent-Context Leakage Control

Before interpreting the original WikiText membership results as fine-tuning memorization, an additional control was introduced.

A concern was that WikiText chunks adjacent to one another may share vocabulary, topic, entities, style, or local context. If so, an apparent membership signal could partially reflect dataset/context differences rather than exact fine-tuning membership.

A new control dataset was therefore constructed containing:

- 5,000 exact members
- 5,000 adjacent nonmembers
- 5,000 random nonmembers

The adjacent examples were selected from the reconstructed WikiText token stream as the closest unused 256-token chunks to each member.

Neighbor quality:

- Immediate-neighbor fraction: `0.9996`
- Mean absolute offset: `1.0004` chunks
- Maximum absolute offset: `2` chunks

Because the original preparation procedure did not retain WikiText article boundaries, this is an **adjacent-context control**, not a guaranteed same-article control.

### Adjacent vs. Random Nonmembers

If ordinary WikiText locality were responsible for most of the membership signal, adjacent and random nonmembers should themselves be strongly distinguishable.

Instead, their AUCs were close to chance:

| Feature | AUC |
|---|---:|
| NLL | 0.5100 |
| Std token loss | 0.5171 |
| Min-K 10% | 0.5128 |
| Min-K 20% | 0.5116 |
| Min-K 30% | 0.5112 |
| Min-K 40% | 0.5109 |

This indicates a small contextual/locality effect, but not one large enough to explain the complete member/nonmember signal.

### Exact Members vs. Adjacent Nonmembers

Exact members remained distinguishable even when compared against immediately adjacent WikiText chunks:

| Feature | Mean AUC |
|---|---:|
| NLL | 0.5327 |
| Std token loss | 0.5438 |
| Min-K 10% | 0.5449 |
| Min-K 20% | 0.5434 |
| Min-K 30% | 0.5406 |
| Min-K 40% | 0.5386 |

This strengthened the interpretation that the measured signal reflects **exact fine-tuning membership**, rather than simply broad WikiText vocabulary or neighboring context.

The control does not prove that all dataset effects have been eliminated, but it makes the simple "members are only distinguishable because they come from different WikiText contexts" explanation substantially less plausible.

---

## Raw-Soft Student Defense

The first defense implemented was an intentionally simple **raw-soft knowledge-distillation baseline**.

The purpose of beginning with an unfiltered student was to establish what ordinary distillation alone does to membership leakage before introducing confidence filtering, clipping, temperature adjustment, anonymization, or other defense mechanisms.

The student:

1. Starts from a fresh Gemma 3 4B PT base model.
2. Uses a fresh LoRA adapter.
3. Learns from the teacher's cached next-token probability distributions.
4. Does not directly optimize against the original hard next-token labels as its primary target.
5. Uses the same 5,000-sequence distillation corpus corresponding to the teacher's members.

This allows the experiment to ask whether transferring the teacher's behavior through soft outputs transmits the same membership signal as directly fine-tuning on the original examples.

### Seed-42 Raw-Soft Training Result

Training completed successfully:

- Training time: `33.31 minutes`
- Peak allocated VRAM: `9.03 GB`
- Initial validation NLL: `2.581132`
- Final validation NLL: `2.246518`

The original Seed-42 realistic LoRA teacher had a validation NLL of approximately `2.24535`.

Therefore, the raw-soft student retained essentially the same held-out language-model utility:

- Teacher validation NLL: approximately `2.24535`
- Raw-soft validation NLL: `2.246518`
- Difference: approximately `+0.00117`

This is important because a privacy improvement obtained by simply destroying model quality would not constitute a useful defense.

---

## Direct Membership Leakage After Raw-Soft Distillation

The raw-soft student was evaluated against the same frozen target attack population:

- 5,000 members
- 5,000 nonmembers

The same token-level membership statistics used for the undefended teacher were calculated.

### Seed-42 Comparison

| Metric | Undefended Teacher AUC | Raw-Soft Student AUC | Change |
|---|---:|---:|---:|
| NLL | 0.532789 | 0.520680 | -0.012109 |
| Perplexity | 0.532789 | 0.520680 | -0.012109 |
| Median token loss | 0.521164 | 0.513272 | -0.007892 |
| Min token loss | ~0.504 | 0.510965 | small increase |
| Max token loss | 0.528975 | 0.517055 | -0.011920 |
| Std token loss | 0.547462 | 0.530106 | -0.017356 |
| Min-K 10% | 0.545011 | 0.528575 | -0.016436 |
| Min-K 20% | 0.542800 | 0.527308 | -0.015493 |
| Min-K 30% | 0.540092 | 0.525380 | -0.014713 |
| Min-K 40% | 0.538160 | 0.523964 | -0.014196 |

The strongest teacher-side direct signals—standard deviation of token loss and Min-K—both moved substantially toward random chance after distillation.

This pattern is more informative than a reduction in a single metric because the major leakage statistics moved in the same direction.

At the same time, the student retained essentially unchanged validation NLL.

### Current Interpretation

For Seed 42, raw-soft distillation therefore shows a promising privacy/utility tradeoff:

- held-out NLL is essentially unchanged;
- NLL-based membership separability decreased;
- difficult-token/Min-K separability decreased;
- token-loss variance separability decreased.

However, these direct attacks are **not sufficient to conclude that the defense is successful**.

A reduction in direct AUC could occur because distillation genuinely removes membership information, but it could also occur because the defense changes the distribution of attack features in a way that makes an attacker trained against ordinary LoRA models less effective.

This motivated a defense-aware adaptive attack.

---

## Threat-Model Clarification

The defended model exposed to an attacker is the **student**, not the private teacher.

The target teacher is used only by the defender to construct the released student.

A realistic black-box attacker therefore observes:

```text
Released Target Student
        ↓
student predictions / token probabilities
        ↓
membership features
        ↓
membership inference
```

The attacker is not assumed to possess the defender's private teacher.

For a Shokri-style attack, however, an attacker may train its **own surrogate models**. If the defense algorithm is known, a strong attacker can reproduce the defense on those surrogate models.

This gives two useful attacker knowledge conditions.

---

## Non-Adaptive Shadow Attack

The first defense evaluation uses ordinary, undefended LoRA shadow models.

```text
Ordinary Shadow LoRAs
        ↓
shadow output features
        ↓
Shokri attack classifier
        ↓
Defended Target Student
```

This represents an attacker whose attack model was developed against ordinary LoRA fine-tuning and is then transferred directly to the defended student.

Most of the infrastructure for this condition already exists from the original MIA experiments. The existing Seed-42 ordinary shadow observations can be used to train the attacker, while the already-generated raw-soft target `membership_scores.csv` supplies the evaluation population.

This experiment measures **attack transfer across the defense-induced distribution shift**.

---

## Defense-Aware / Adaptive Shadow Attack

A stronger attacker is assumed to know that the released target was produced using raw-soft distillation.

The attacker still does **not** possess the target teacher.

Instead, it trains its own ordinary shadow teachers and reproduces the known defense:

```text
Attacker's Shadow Teacher
        ↓
Raw-Soft Distillation
        ↓
Attacker's Shadow Student
        ↓
query Shadow Student
        ↓
membership features
        ↓
train Shokri classifier
        ↓
query released Target Student
```

This tests whether membership leakage remains exploitable after the attacker adapts its attack methodology to the defense.

### Adaptive Shadow Pipeline

The six previously trained Seed-42 shadow LoRAs are currently being distilled into six raw-soft shadow students.

For each shadow:

1. Load the existing independently trained shadow-teacher LoRA.
2. Use that shadow's own 5,000 known member examples.
3. Cache the shadow teacher's Top-64 next-token distributions.
4. Initialize a fresh Gemma 3 4B PT model with a fresh LoRA adapter.
5. Train a raw-soft shadow student from the shadow teacher outputs.
6. Query the **shadow student**, not the teacher, on:
   - 5,000 known members
   - 5,000 known nonmembers
7. Extract the same membership features used against the target.

Each defended shadow therefore contributes 10,000 labeled attack observations.

Across six shadows:

- Members: 30,000
- Nonmembers: 30,000
- Total adaptive attack-training observations: 60,000

The teacher-output caches are used only to reproduce the defense. They are **not** supplied as features to the final membership attacker.

The final adaptive attack classifier will therefore be trained entirely from features derived from **defended shadow-student outputs** and evaluated entirely from features derived from the **defended target-student output**.

### Resume-Safe Implementation

The adaptive pipeline was implemented with separate parameterized stages for:

- teacher Top-K caching;
- raw-soft shadow-student training;
- shadow-student membership scoring;
- final aggregation.

The runner detects completed stages and skips them when restarted.

This was necessary because six Gemma shadow distillations represent a multi-hour GPU workload, and a failure late in the run should not require retraining previously completed shadows.

The current Seed-42 adaptive run was relaunched under `nohup` after the first shadow teacher cache completed successfully.

The resumed process correctly detected:

```text
[SKIP] cache exists:
data/gemma/defense/seed_42/raw_soft/shadows/shadow_00/teacher_topk/teacher_topk_cache.pt
```

and proceeded directly to training the `shadow_00` raw-soft student.

At the time of this progress update, the adaptive shadow pipeline is still running. **No adaptive MIA result should be reported until all six defended shadow students have completed and the new attack classifier has been trained.**

---

## Planned Three-Way Attack Comparison

Once adaptive shadow generation completes, the defense will be evaluated under three directly comparable conditions:

| Condition | Attacker Training Models | Evaluation Target |
|---|---|---|
| Original MIA | Ordinary LoRA shadows | Undefended realistic LoRA |
| Non-adaptive defense MIA | Ordinary LoRA shadows | Raw-soft student |
| Adaptive defense MIA | Raw-soft distilled shadow students | Raw-soft student |

This comparison is necessary to distinguish genuine privacy improvement from simple attacker mismatch.

Possible interpretations:

**Non-adaptive attack decreases, adaptive attack recovers the original leakage**

The defense primarily changes the attack-feature distribution. An attacker aware of the defense can compensate for that shift.

**Both non-adaptive and adaptive attacks remain substantially closer to chance**

This is stronger evidence that distillation removes or suppresses exploitable fine-tuning membership information.

**Adaptive attack partially recovers leakage but remains below the undefended baseline**

This would indicate that distillation provides a real but incomplete privacy benefit: knowledge of the defense helps the attacker recover some signal, but does not fully restore the leakage available from the undefended teacher.

No conclusion among these possibilities will be selected until the adaptive experiment finishes.

---

## Why the Experimental Design Changed

Several changes were introduced as the Gemma experiment developed.

### Added multiple seeds

The original result could have reflected a favorable random initialization or dataset/model-training fluctuation.

The realistic target and Shokri attack were therefore replicated across four seeds:

- 42
- 123
- 456
- 789

The repeated above-chance Shokri AUC established that the original leakage result was not unique to one run.

### Added adjacent-context controls

WikiText is reconstructed into contiguous token chunks. This raised the possibility that vocabulary/topic/context locality could masquerade as membership leakage.

The adjacent-context experiment was added to explicitly test this alternative explanation.

### Began with raw-soft distillation

Rather than immediately introducing confidence thresholds or other mechanisms, an unfiltered soft-label student was trained first.

This provides a clean control answering:

> How much privacy benefit comes from distillation itself?

Any later filtered/clipped/temperature-based defense can therefore be compared against ordinary distillation rather than incorrectly attributing all improvement to the additional mechanism.

### Added adaptive shadows

Direct attacks and an ordinary shadow attacker may underestimate leakage if the defense changes the distribution of membership features.

The adaptive shadow pipeline was therefore added so that the attacker is allowed to know the defense procedure and reproduce it using its own surrogate models.

This creates a substantially stronger test of whether the defense actually suppresses membership information.

---

## Current Status

Completed:

- Realistic Gemma 3 4B LoRA target training
- Direct membership attacks
- Six-shadow Shokri-style MIA
- Four-seed replication
- Adjacent-context membership control
- Seed-42 teacher Top-K cache
- Seed-42 raw-soft student training
- Seed-42 raw-soft direct membership scoring
- Raw-soft adaptive shadow pipeline implementation

Currently running:

- Six Seed-42 raw-soft adaptive shadow students
- Shadow-student membership scoring and 60,000-observation aggregation

Next:

1. Finish all six raw-soft shadow students.
2. Train logistic-regression and random-forest Shokri attackers on the 60,000 defended-shadow observations.
3. Evaluate those attackers on the frozen Seed-42 raw-soft target population.
4. Run the non-adaptive ordinary-shadow attacker against the same raw-soft target population.
5. Compare original, non-adaptive-defense, and adaptive-defense MIA results.
6. If the Seed-42 defense result survives the adaptive attack, replicate the defense across Seeds 123, 456, and 789.
7. Only after establishing the raw-soft baseline should additional confidence-aware filtering/clipping/temperature variants be evaluated.

The central methodological principle remains:

> **A defense should be credited with reducing membership leakage only if the reduction survives an attacker reasonably adapted to the defended model while useful model performance is retained.**

---

## Seed-42 Raw-Soft Adaptive MIA Results

The six-shadow raw-soft adaptive pipeline completed successfully.

The final adaptive shadow attack dataset contained:

- 6 independently trained shadow teachers
- 6 corresponding raw-soft distilled shadow students
- 5,000 known members per shadow student
- 5,000 known nonmembers per shadow student
- 30,000 total member observations
- 30,000 total nonmember observations
- 60,000 total attack-training observations

The combined shadow-student attack dataset was saved to:

`results/gemma/defense/seed_42/raw_soft/shadows/all_shadow_membership_scores.csv`

The frozen defended target evaluation population remained:

- 5,000 target members
- 5,000 target nonmembers
- 10,000 total target observations

This target population was not used to fit the membership classifier.

### Adaptive Attack Procedure

The adaptive attack used the same nine output-derived features as the original Shokri-style MIA:

- NLL
- median token loss
- minimum token loss
- maximum token loss
- standard deviation of token loss
- Min-K 10%
- Min-K 20%
- Min-K 30%
- Min-K 40%

Two attack classifiers were trained on the 60,000 defended-shadow observations:

1. StandardScaler + Logistic Regression
2. Random Forest

The attack was deliberately **defense-aware**. Each attack shadow reproduced the raw-soft distillation process before membership features were collected.

Importantly, the attacker did not receive the target teacher, target teacher outputs, or target membership labels during attack training. The target membership labels were used only after prediction to calculate evaluation metrics.

---

## Adaptive Raw-Soft Attack Results

### Logistic Regression

| Metric | Result |
|---|---:|
| Accuracy | 0.521700 |
| Precision | 0.521113 |
| Recall | 0.535600 |
| F1 | 0.528257 |
| ROC-AUC | **0.523833** |

### Random Forest

| Metric | Result |
|---|---:|
| Accuracy | 0.522900 |
| Precision | 0.520799 |
| Recall | 0.573400 |
| F1 | 0.545835 |
| ROC-AUC | **0.526652** |

The adaptive attack therefore remained slightly above random guessing, demonstrating that the raw-soft student still contains some exploitable membership information.

However, the adaptive attacker recovered substantially less membership separability than was observed against the undefended Seed-42 realistic LoRA.

---

## Undefended vs. Raw-Soft Adaptive Attack

The appropriate Seed-42 comparison is:

| Target | Shadow Models | Logistic AUC | RF AUC |
|---|---|---:|---:|
| Undefended realistic LoRA | Undefended LoRA shadows | **0.545891** | **0.544778** |
| Raw-soft student | Raw-soft distilled shadows | **0.523833** | **0.526652** |

Absolute AUC changes:

- Logistic: `0.545891 -> 0.523833`
  - reduction = **0.022058**
- Random Forest: `0.544778 -> 0.526652`
  - reduction = **0.018126**

Because an AUC of 0.50 represents random ranking, another useful diagnostic is the amount of AUC above random chance.

### Excess AUC Above 0.50

Logistic regression:

- Undefended: `0.545891 - 0.50 = 0.045891`
- Raw-soft: `0.523833 - 0.50 = 0.023833`
- Reduction in excess AUC: approximately **48.1%**

Random forest:

- Undefended: `0.544778 - 0.50 = 0.044778`
- Raw-soft: `0.526652 - 0.50 = 0.026652`
- Reduction in excess AUC: approximately **40.5%**

This excess-AUC calculation is used only as a descriptive measure of how much above-chance separability was removed. The raw ROC-AUC values remain the primary reported attack metric.

---

## Utility/Privacy Result

The privacy reduction occurred with essentially unchanged held-out language-model utility.

Previously measured validation NLL:

| Model | Validation NLL |
|---|---:|
| Realistic LoRA teacher | ~2.24535 |
| Raw-soft student | 2.246518 |

Difference:

`~+0.00117 NLL`

This corresponds to approximately a 0.05% relative increase in validation NLL.

The Seed-42 experiment therefore provides evidence that raw-soft distillation can reduce membership leakage without requiring a substantial degradation in held-out model performance.

The result is particularly important because the reduction survived an attacker that was explicitly adapted to the defense.

---

## Relationship to Direct Membership Signals

The adaptive attack result is consistent with the previously observed reductions in individual membership statistics.

Earlier direct Seed-42 measurements included:

| Feature | Undefended AUC | Raw-Soft AUC |
|---|---:|---:|
| NLL | 0.532789 | 0.520680 |
| Median token loss | 0.521164 | 0.513272 |
| Max token loss | 0.528975 | 0.517055 |
| Std token loss | 0.547462 | 0.530106 |
| Min-K 10% | 0.545011 | 0.528575 |
| Min-K 20% | 0.542800 | 0.527308 |
| Min-K 30% | 0.540092 | 0.525380 |
| Min-K 40% | 0.538160 | 0.523964 |

The adaptive Shokri attack confirms that these reductions cannot be explained solely by evaluating individual attack statistics independently.

Even after allowing the attacker to combine multiple features learned from defended shadow students, attack AUC remained substantially below the corresponding undefended Seed-42 attack.

---

## Adaptive Random-Forest Feature Importance

The adaptive random forest produced the following feature importances:

| Feature | Importance |
|---|---:|
| Max token loss | 0.143158 |
| Std token loss | 0.125965 |
| Min-K 10% | 0.114368 |
| Median token loss | 0.114136 |
| Min-K 20% | 0.106526 |
| NLL | 0.104212 |
| Min-K 30% | 0.100377 |
| Min-K 40% | 0.099689 |
| Min token loss | 0.091570 |

No single feature dominated the adaptive random forest.

Maximum token loss and token-loss standard deviation were the most important features, while Min-K statistics, median loss, and NLL also contributed meaningful portions of the attack model.

This is useful because the reduced adaptive AUC does not appear to result simply from the disappearance of one specific attack feature. The attacker was able to use a broad collection of output statistics but still recovered only modest membership separability.

---

## Current Interpretation

The Seed-42 raw-soft experiment now supports a stronger conclusion than the initial direct-score experiment.

The initial direct attack showed that distillation moved several individual membership signals toward random chance.

The adaptive experiment additionally shows that:

1. An attacker can reproduce the known defense using independently trained shadow teachers.
2. The attacker can distill those teachers into defended shadow students.
3. The attacker can train directly on the output behavior of those defended shadow students.
4. The resulting attack still performs substantially worse against the defended target than the matched original attack performed against the undefended target.

Therefore, the observed reduction cannot be attributed solely to using an attacker trained on the wrong model type.

For Seed 42, the evidence currently supports the following statement:

> **Raw-soft knowledge distillation preserved held-out language-model utility while reducing fine-tuning membership separability, and the reduction persisted under a defense-aware Shokri-style shadow-model attack.**

This should currently be treated as a **Seed-42 result**, not yet as a general multi-seed defense conclusion.

The undefended experiment has already been replicated across four seeds, but the raw-soft defense has not yet received the same multi-seed replication.

---

## Important Threat-Model Distinction

Two defended-target attacks are now intentionally distinguished.

### Non-Adaptive Attack

The attacker trains on ordinary, undefended LoRA shadow models and attacks the raw-soft target student.

This measures how well an ordinary MIA transfers to a model whose training procedure has changed.

### Adaptive Attack

The attacker knows the raw-soft defense procedure and reproduces it using its own shadow models.

The attack classifier is consequently trained on **raw-soft shadow-student outputs** before attacking the raw-soft target student.

The completed result above is the **adaptive attack**.

This is the stronger of the two evaluations because the attacker is not penalized for being unaware of the defense.

The non-adaptive attack remains useful as a comparison because it will quantify how much membership information the adaptive attacker recovers by explicitly modeling the defense.

---

## Next Experiment: Non-Adaptive Attack Transfer

The next experiment will train the same logistic-regression and random-forest membership classifiers using the original undefended shadow-model observations and evaluate them against the exact same frozen Seed-42 raw-soft target population.

No new target model or target scoring run is required.

The intended comparison will therefore be:

| Experiment | Attack Training Population | Target |
|---|---|---|
| Original MIA | Undefended shadows | Undefended target |
| Non-adaptive defense MIA | Undefended shadows | Raw-soft target |
| Adaptive defense MIA | Raw-soft shadows | Raw-soft target |

The adaptive result is now known:

- Logistic AUC: **0.523833**
- Random Forest AUC: **0.526652**

The non-adaptive result remains to be measured.

If the non-adaptive attack approaches random chance while the adaptive attack reaches approximately 0.524-0.527, this would demonstrate that adapting to the defense recovers some membership information.

If that occurs while the adaptive result remains below the original undefended attack, it would support the interpretation that raw-soft distillation produces both:

1. an attack-distribution shift that harms an unaware attacker, and
2. a genuine reduction in membership separability that remains after the attacker adapts.

---

## Updated Raw-Soft Defense Status

Completed:

- Seed-42 teacher Top-64 cache
- Seed-42 raw-soft target student
- Raw-soft target utility evaluation
- Raw-soft direct membership scoring
- Six defense-aware shadow-teacher caches
- Six raw-soft shadow-student training runs
- Six shadow-student membership-scoring runs
- 60,000-example adaptive shadow attack dataset
- Adaptive logistic-regression MIA
- Adaptive random-forest MIA
- Adaptive feature-importance analysis

Current key Seed-42 result:

- Undefended Logistic AUC: **0.545891**
- Adaptive Raw-Soft Logistic AUC: **0.523833**
- Undefended RF AUC: **0.544778**
- Adaptive Raw-Soft RF AUC: **0.526652**
- Teacher validation NLL: **~2.24535**
- Raw-soft validation NLL: **2.246518**

Next:

1. Run the non-adaptive undefended-shadow -> raw-soft-target attack.
2. Compare original, non-adaptive, and adaptive attack conditions.
3. Freeze the raw-soft defense design.
4. Replicate the defended experiment across Seeds 123, 456, and 789.
5. Aggregate mean and standard deviation across defense seeds.
6. Only after raw-soft replication, proceed to additional confidence-aware defense variants such as temperature modification, clipping, or confidence filtering.

The defense should not be considered validated from Seed 42 alone. Multi-seed replication remains necessary before making a general claim about the effectiveness of raw-soft distillation.