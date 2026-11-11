# Exemplar matcher: evidence, method, and open questions

Paths and commands below are relative to `app/` unless they refer to documentation.

Last reviewed: 10 September 2026.

Status: the runtime method is implemented and unit-tested.

**Measurement scope.** Every number in this document was produced in a CPU
CPU pilot, on public corpora (the "August" numbers on 21 August
2026, the "Round 3" numbers on 2 September 2026, [round-3.md](round-3.md)).
None of them is a project measurement: these runs did not use project
hardware. They explain why the shipped recipe looks the way it
does. The numbers the project reports come from recorded runs of
`scripts/matcher/train_projection.py --eval` and `scripts/matcher/evaluate.py`
on TORGO under `reports/`, which replace the figures below as they are
made.

## 1. Research question

Can a short, enrolled word or phrase be recognized from a small number of a
speaker's own recordings when an open-vocabulary recognizer is unreliable?

The matcher solves a narrower task:

```text
input: one utterance
enrolled set: a finite bank of labeled recordings
output: ranked enrolled labels, or rejection when calibrated
```

It is not a sentence recognizer, pronunciation dictionary, intent predictor,
or general correction layer. If the spoken content is outside the enrolled
set, an uncalibrated matcher still returns the nearest label, and that limits
where it is safe to use.

## 2. Prior work

Lea et al.'s Latent Phrase Matching (LPM) is the closest published system. LPM
enrolled at least two examples per phrase, represented audio with a compact
keyword embedding network, compared sequences with dynamic time warping, and
used phrase-specific thresholds. It was evaluated on an internal dataset of
32 people with dysarthria and on EasyCall.

Key observations from that paper:

- phrase matching outperformed a general ASR baseline for moderate and severe
  speech;
- performance degraded as the number of enrolled phrases grew;
- longer phrases were generally more distinguishable;
- different-day and microphone placement affected accuracy;
- lower SNR harmed both matching and ASR;
- open-set thresholds were needed because the nearest phrase is not always
  present.

These findings support constrained matching but do not show that DTW is always
the best scoring rule or that one global threshold transfers. Primary source:
[Lea et al., Latent Phrase Matching for Dysarthric Speech](https://arxiv.org/abs/2306.05446).

WavLM provides the frozen representation used here. It was trained as a broad
self-supervised speech model with masked prediction and denoising objectives.
Its representations contain linguistic, speaker, and acoustic information;
the matcher therefore needs to suppress nuisance variation rather than assume
the embedding is already lexical. Source:
[Chen et al., WavLM](https://arxiv.org/abs/2110.13900).

## 3. Current pipeline

```text
16 kHz waveform
  -> trim edge silence
  -> peak normalize
  -> WavLM-base-plus hidden layer 9
  -> pool across frames: mean and standard deviation (1,536 dimensions)
  -> subtract session mean (enrolment: its session's mean; live: the shrunk running mean)
  -> center and apply learned 1,536 x d projection
  -> (a d x d session transform is loadable but not shipped -- measured, lost)
  -> length normalize
  -> one prototype per label = mean of its takes; cosine against every prototype
  -> best score per label
  -> optional exact Parakeet text vote
  -> top-k labels and margin (no rejection: the nearest label is always named)
```

The implementation is split so numerical behavior can be tested without
downloading WavLM:

- `matcher/engine.py`: bank lifecycle, embedding, projection, scoring orchestration;
- `matcher/scoring.py`: normalization, cosine, label aggregation, rank and margin;
- `matcher/training.py`: LDA, WCCN, NAP, and research diagnostics;
- `scripts/matcher/train_projection.py`: projection fitting and held-out-speaker evaluation;
- `scripts/matcher/evaluate.py`: target-corpus closed-set and cross-session protocols.

## 4. Representation choice

The matcher uses WavLM-base-plus layer 9, pooled to the frame mean and the
frame standard deviation. The historical TORGO layer probe (August, mean
pooling only) recorded the following one-shot top-1 accuracies:

| Layer | 0 | 3 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WavLM base-plus | .213 | .208 | .239 | .248 | .264 | .278 | **.294** | .282 | .268 | .284 |
| HuBERT base | .217 | .213 | .269 | .278 | .270 | .269 | **.283** | .272 | .258 | .249 |

These figures are the August CPU pilot measurements, not published
benchmark results and not project measurements. Round 3 repeated the sweep with the projection in place, on both the
one-shot and the cross-session protocols: layer 9 still wins (L8 .442/.399,
L9 .452/.384, L10 .445/.333, L7 .417/.327, L12 .429/.350). Concatenating
layers 8-10 looked like +2 on the full vocabulary and was worth nothing on
words the projection had not seen -- memorised vocabulary, not adopted.

Pooling deliberately discards frame alignment, which favors overall
lexical content and robustness over explicit timing. Round 3 tested
the order-preserving alternative (3- and 5-segment pooling) and it lost badly
(.326 and .283 against .452 one-shot), the same lesson DTW taught in August.
Adding the frame standard deviation to the mean, which keeps no order, gained
+3.6 cross-session on unseen words (§6.3 of round-3.md).

## 5. Projection

Raw WavLM vectors contain lexical identity together with speaker, microphone,
session, and condition. The packaged projection uses linear tools from
discriminant analysis and speaker verification.

### 5.1 Shrinkage LDA

The 1 October method review corrected the between-class scatter: a label with
one example contributes its class mean even though its within-class scatter is
zero. Previously such labels were omitted from both terms. Future projection
fits use the corrected calculation; the packaged projection and historical
scores were not regenerated. Compare old and new fits on the same held-out
speakers before claiming a recognition gain from this correction.

Let `x` be a 1,536-dimensional, session-centered embedding. LDA estimates
within-label scatter `Sw` and between-label scatter `Sb`, then finds directions
with high between-label variation relative to within-label variation:

```text
Sb w = lambda Sw w
```

Because the feature dimension can be large relative to the corpus, `Sw` is
shrunk toward a scaled identity before inversion. `matcher.training.lda_fit` solves the
generalized problem through whitening and a symmetric eigendecomposition.
The output is `W` with shape `1,536 x d`, normally `d = 128`.

How `Sw` is estimated turned out to be the single largest lever in three
rounds of work. The August projection used scikit-learn's per-class shrinkage,
which with 2-10 takes per word over-regularises each class covariance toward
identity; one covariance pooled over all words and shrunk 10% toward identity
(`alpha = 0.1`, what `lda_fit` does) lifts layer-9 one-shot top-1 .452 -> .502
and cross-session .384 -> .454 on TORGO (round-3.md §6.1). Too little
shrinkage (Ledoit-Wolf's automatic 0.002) also loses; 0.05-0.2 is a broad
optimum.

Training classes are word labels pooled across training speakers and sessions.
This asks the projection to keep recurring word differences and discount
directions that vary inside a word class.

### 5.2 WCCN

Within-Class Covariance Normalization estimates variation among repeated
takes of the same `(speaker, word)` class and whitens that covariance. In the
project convention:

```text
P = (x - mean) @ W
T T^T approximates inverse(within-class covariance)
output = P @ T
```

This makes a unit of distance more comparable across directions. It only has
a defensible interpretation when the fitting data contains repeated labels
and, preferably, multiple sessions.

### 5.3 NAP

Nuisance Attribute Projection estimates directions along which session means
move for the same `(speaker, word)` class and removes the top `k` directions:

```text
T = I - V V^T
```

NAP is simpler and more destructive than WCCN. Increasing `k` can reduce
session information while also deleting lexical information. Choose `k` on a
held-out protocol, never on the training fit.

### 5.4 Required diagnostics -- and the Round 3 verdict

Round 3 ran these diagnostics on TORGO (round-3.md §4). WCCN was worth +1.7 /
+2.8 with the August covariance estimate and *cost* 1-3 points once the
estimate was fixed; NAP was worth nothing at any rank; unsupervised CORAL was
worth nothing; and the "which session?" probe stayed at .79-.81 (chance .61)
under every transform learned from other speakers, against .51 after
per-session mean subtraction on the speaker's own data. Session drift is
speaker-specific: no transform learned from other speakers removes it. The
shipped projection has no `T`. A session transform earns its place only if:

1. cross-session label accuracy improves or holds;
2. NAP's session-identification probe moves toward chance; WCCN is invertible
   and is not expected to erase session information;
3. gains appear on held-out speakers or held-out sessions;
4. no single large speaker dominates the aggregate;
5. the transformation remains numerically stable.

The current synthetic tests verify the algebra. They do not substitute for
real cross-session speech.

## 6. Enrollment and scoring choices

### Keep every take; score against the prototype

The bank keeps every take (they feed the session vectors and the per-take
detail in the score fields), but the label's *decision* score is the cosine to
its prototype -- the mean of its takes' unit vectors, re-normalised. Round 2
argued the opposite ("distinct productions represent different valid
articulations; never average"). Round 3 measured it on EasyCall, 31 dysarthric
speakers with six sessions each (round-3.md §7.2): with the projection in
place the prototype beats max-over-takes at every enrolment size, by 2.6
points at five sessions and by 4 on the severe speakers; the top-2 mean and a
mean/max fusion do not beat the plain mean. In the projected space averaging
cancels take-to-take noise, which is what the projection was fitted to
discount.

The same experiment gives the enrolment curve the recording plan follows: each
additional session adds about five points with no plateau at five (.547 ->
.786 top-1; severe .333 -> .554), and the most recent session alone beats the
oldest alone by eight points. Evaluation must therefore report accuracy and
rejection as a function of sessions, not only of takes.

### Session centering

Each enrolled session's raw embeddings are centered by that session's mean.
The legacy evaluation's session-normalized/projected rows also use each complete
test session's mean, including later clips. That is an offline diagnostic and
can overstate what is available while listening. `scripts/matcher/evaluate.py`
now reports a separate causal cross-session row: all enrollment takes form the
label prototypes; each test session starts with the enrollment bank's prior,
then uses only the current and earlier queries to update its mean. Input order
must represent recording order when interpreting it as a live-session replay.
The legacy one-take comparison and the causal row also differ in enrollment
count, so their gap does not isolate centering alone. Counts include query uses
across enrollment folds, not independent recordings. Unknown-label queries
update the causal mean but are excluded from closed-set accuracy. This still
does not test rejection, ASR voting, stop/sleep/wake actions or real microphone
segmentation.
At runtime, a running mean estimates the current session offset, shrunk toward
the bank's overall mean by `n / (n + n0)` with `n0 = 20`
(`WILLSPEAK_MATCH_SESSION_WARMUP`). Round 3 §6.5 measured why: the unshrunk
running mean subtracts the first utterances from themselves and costs seven
points of top-1 in the first ten utterances of a session; shrunk, it never
hurts and is worth about one point overall. Most of what "session
normalization" is worth (+3 cross-session) is the enrolment-side centering,
which is free. Using the fixed probe list to estimate the day's shift instead
(5, 10 or 20 anchor words) was also measured and hurts at every count: a
few-take estimate of the shift is noisier than the shift itself.

### Nearest-session weighting (removed)

An earlier runtime could add a small score bonus for exemplars from
sessions acoustically similar to the live one. The weight was always zero,
the bonus was applied to per-take scores while ranking used prototypes (so it
could never change a decision), and it was removed during code cleanup. It stays here as
a recorded idea: with only one enrolled session it cannot choose among days,
and with two the test session must not leak its labels into enrolment.

### Exact sentence-ASR vote

If the normalized Parakeet transcript exactly equals an enrolled label, the
matcher can add a small vote to that label. The gate is exact by design. Fuzzy
text correction can pull an unrelated or out-of-vocabulary utterance toward a
known label and is not part of the current scoring policy.

## 7. Historical results (CPU pilot runs, August 2026)

The August 2026 report recorded a leave-one-speaker-out experiment on eight
TORGO dysarthric speakers with vocabularies of roughly 78-296 words. Its main
reported macro results were:

| Condition | Top-1 | Top-3 | Interpretation at the time |
|---|---:|---:|---|
| raw one-shot cosine | .28 | not preserved in the summary table | baseline |
| projected one-shot cosine | .42 | not preserved in the summary table | about 50% relative top-1 gain |
| synthetic exemplars after projection | about .50 | about .72 | day-zero research condition |

The report also recorded:

- cross-session top-1 improving from .185 to .377 after projection;
- same-take/different-microphone accuracy improving from .41 to .635;
- open-set AUROC improving from .65 raw to .74 projected;
- a proxy corpus improving from .72/.85 raw top-1/top-3 to .87/.95 projected
  with one take per label;
- two and three takes producing further gains in that proxy corpus.

These measurements justify keeping the projection research, with these
caveats:

- TORGO is lab-recorded and small;
- vocabulary overlap and repeated read prompts simplify parts of the task;
- some per-speaker test sets are small;
- the proxy corpus is not dysarthric speech;
- synthetic exemplars are not in the current live enrollment path;
- no current target-corpus rerun accompanies this cleanup;
- AUROC .74 is not strong enough to assume safe open-set rejection.

The packaged `projection.npz` is an artifact of that experiment, but it does
not prove the experiment is reproducible. Keep
the source embedding manifests, fitting command, commit, and output table when
the next projection is trained.

## 8. Negative results retained from the earlier study

These are written down so ideas that already failed under this protocol do
not get rebuilt.

### Dynamic time warping

The earlier study recorded that DTW improved over raw mean-pooled cosine but
lost to the learned projection and regressed when fused with projected scores.
That result differs from LPM, where DTW was central. The discrepancy may arise
from representation, phrase length, enrollment size, or session conditions.
It should be described as a project result under one protocol, not a general
claim that DTW is ineffective.

### Raw duration

Adding utterance duration as a score did not help in the earlier study. More
structured prosody or relative timing could still help; raw duration is just
not part of the live score.

### Averaging and donor weighting

The earlier experiments favored keeping takes rather than collapsing them and
did not benefit from down-weighting synthetic exemplars. Synthetic enrollment
is no longer a shipped path, so that latter result is research history rather
than current behavior.

### Contrastive refinement

A refined projection reportedly improved TORGO and regressed the proxy corpus,
suggesting corpus-specific overfitting. The repository keeps the simpler
linear projection until a stronger method wins on an external held-out
distribution.

## 9. Open-set rejection (not implemented)

Closed-set accuracy asks whether the correct enrolled label ranks first. Open-
set rejection asks whether the spoken content belongs to the enrolled set at
all. A system can be strong at the first and unsafe at the second.

The runtime has no rejection. It always returns the nearest enrolled label,
and the only thing standing between that label and the loudspeaker is the
policy layer: `vocab_override` is off by default, and when it is on the label
must clear a score threshold and a margin over the runner-up. An earlier version of
the engine carried an adaptive symmetric score normalisation (AS-norm)
against a cohort of impostor vectors; no cohort was ever built for any
speaker, the `rejected` flag it produced was never read by the policy layer,
and the whole path was removed during code cleanup.
In the CPU pilot the design reached AUROC .74 (August) and about .85 only
from the third enrolled day (Round 3, EasyCall), not strong enough to trust.

If rejection is revisited, the cohort must be disjoint from enrolled labels
and evaluation items, and a threshold must be selected on development data
and then frozen. Report:

- AUROC and precision-recall curve;
- false-accept rate at the chosen threshold;
- true-accept rate at that threshold;
- results by phrase length and session;
- performance as enrollment size grows;
- noise and silence false accepts.


## 10. Evaluation protocols

### Closed vocabulary, random takes

For each label with at least `k + 1` recordings, enroll `k`, test the remaining
takes, repeat with a fixed random seed, and report top-1/top-3. Compare raw,
session-centered, projected, and projection-without-session-transform spaces.

This protocol estimates the value of extra takes but can mix days between
enrollment and test.

### Cross session

Enroll from one session and test matching labels from all other sessions;
repeat for each possible enrollment session. Report macro per-session and
micro aggregate counts. This is the principal deployment-like protocol.

### Open set (future)

If rejection returns: partition labels into enrolled, impostor, and cohort
groups with a stable mapping. Enroll one or more takes from enrolled labels;
positive tests are unseen takes of those labels; negatives are takes of
never-enrolled labels.

### Controls

- shuffled labels should fall to chance;
- duplicate audio must never appear on both sides;
- same recording through two file paths counts as leakage;
- exact prompt repeats across splits require session separation;
- fit projections on training speakers only in held-out-speaker evaluation;
- choose thresholds on development, not on final test negatives.

## 11. Runtime safety rules

- Do not let an uncalibrated nearest label replace arbitrary sentence text.
- Keep phrase override off unless thresholds were measured for the speaker and
  microphone.
- Use multiple sessions for control-phrase enrollment.
- Prefer a distinctive multi-syllable stop phrase over a common short word.
- Reset the live session mean after a microphone or day change.
- Rebuild the bank after accepted/rejected status changes.
- Display top candidates and the source of any override in diagnostic views.
- There is no rejection capability; do not describe the matcher as having one.

## 12. Next experiments

Explored in the CPU pilot in Round 3 on TORGO/EasyCall
([round-3.md](round-3.md)): the projection table with the shipped code path
(§6.1); the cross-day enrolment curve (§7.2); the WCCN/NAP held-out
comparison (§4); cold-start live centering (§6.5); the layer sweep (§6.2).

This semester (TORGO; see `docs/measurements.md`):

1. Re-fit the projection with one TORGO speaker held out at a time
   (`train_projection.py --eval`, T-2) and report the raw-versus-projected
   gain on the held-out speakers. The vocabulary is shared across TORGO
   speakers, so this measures transfer across *speakers*, not across words -
   say so on the poster.
2. Closed-set and cross-session accuracy on the TORGO words with that
   projection (`evaluate.py --emb`, T-3).
3. Matcher latency and memory on the Pi 5 (bank rebuild and one live match)
   alongside the Parakeet numbers already measured.
4. Phrase-bank growth from 10 to 25, 50, and 100 labels.

Client phase (after IRB approval):

5. His one-, two-, and three-take curve and his sessions curve after three
   recording days.
6. Refit the projection with consented family recordings of his vocabulary
   added to TORGO, and measure the in-vocabulary gain.
7. Decide whether open-set rejection is needed at all with `vocab_override`
   off; if it is, build and measure it on his own out-of-vocabulary speech.

The matcher graduates from research support to an enabled override only when
the cross-session gate passes on the actual deployment setup.
