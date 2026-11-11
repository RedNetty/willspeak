# Lab test 37: can single words and sentences go to different adapters, with the device telling them apart?

Proposal F-2 (extension). On the common lab set the whole gap to the 35% bar is single words, since sentences are already under it. I count wrong words of the 425 pooled test words, split by clip kind (122 single-word words, 303 sentence words).

Status: written 9 November 2026. Prediction not yet written.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

Stage P is the most accurate system so far: a model trained on the other severe speakers, then a block-8 adapter at 6e-4. It gets 178.0 of 425 wrong (41.9%): 85.7 of 303 sentence words (28.3%, under the 35% line) and 92.3 of 122 single-word words (75.7%). The bar is 148 wrong, so single words would have to drop to 62 or fewer, 30 fewer than now.

Whole-block training from the pretrained model is best on single words (85.0 wrong) and stage P's adapter on sentences (85.7). Using each on its own kind would give about 170.7, but earlier notes dismissed that as "not a device system" because the device doesn't know a clip's kind.

Lab test 36 found a signal the device can see. The first decode has exactly one word for 384 of 393 single-word clips and for only 2 of 120 sentence clips, about 97% accurate. So the device can ask its usual model "did you hear one word?" and use a second model when it did.

Here I test whether that second model can be a small adapter trained only on a speaker's single-word clips, on the same base and block-8 recipe, so no 2.4 GB whole-block file. If the ordinary adapter's first decode has one word, the word adapter's answer is used. Routing is computed afterwards from the saved decodes, with no retraining.

## Setup

Everything starts from stage P's base, `models/parakeet-ft-big-c-pre-<speaker>` (F-1's recipe on the other severe speakers, seed 17). The queue rebuilds it if its weights were cleaned, about 8 minutes a speaker. Every adapter is lab 34's block 8: after encoder block 8, 15 epochs, 6e-4, scoring-form targets, last four epochs averaged, no checkpoint picked on dev (`--select last`). Seeds 17, 18 and 19.

| Arm | Trained on | Run name | Role |
| --- | --- | --- | --- |
| control | every train and dev clip | `lab-37-all-<speaker>-s<seed>` | stage P's recipe, retrained here on the same base |
| word adapter | that speaker's single-word train and dev clips only | `lab-37-word-<speaker>-s<seed>` | the new model |
| sentence adapter | that speaker's sentence train and dev clips only | `lab-37-sentence-<speaker>-s<seed>` | does specialising help sentences too (`routing_arms`, on by default) |

Every adapter decodes all 166 test clips (M01, M02, M04: 122 single-word and 44 sentence clips), so each arm is a full system and also half of a routed one.

The control is retrained, as in lab 34, because a rebuilt base doesn't repeat exactly. Lab 28's retrain of stage P's seed 17 scored 176 against the original 183, and `ensure_pretrained_models` rebuilds a base whenever its weights were cleaned. Comparing against old stage P decodes would mix up base and recipe, so those (`big-p-pre-ad8-lr6e-4-*`, 178.0) are a reference row only. Lab 34's retrained block 8 scored 174.3 (176, 172, 175): 83.3 on sentences, 91.0 on single words.

Routing is done by `scripts/asr/kind_route.py` on CPU from the saved decodes. A clip gets the word adapter's decode if the detector calls it a single word, otherwise the other model's.

| Detector | Sees | Device system? |
| --- | --- | --- |
| first decode has one word | the control's first decode, after the scoring normalisation (an empty decode counts as not one word) | yes: it needs a decode the device makes anyway |
| short clip | clip duration against a cut chosen on that speaker's own labelled train and dev clips, never the test clips | yes |
| oracle | the clip's true kind | no: an upper bound for the routing |

The "other" model is the control or the sentence adapter. Word count always reads the control's first decode, so the sentence-adapter version needs three models. Whether a device with only the word and sentence adapters could read the sentence adapter's own first decode isn't measured. Duration overlaps in lab 36's pools (sentences start at 2.4 s, the 95th percentile of single words is 4.1 s), so word count is the main detector.

When older decodes exist they give reference rows, labelled as older bases. The whole-block model at 8e-5 for 20 epochs (`big-p-pre-wb-lr8e-5-ep20-*`, 192.3 wrong) also serves as a routed word model. That's the 2.4 GB model the adapter is meant to replace. Nothing new is trained for it.

`train.py --kinds word|sentence` already keeps only one kind of train and dev clip (lab K used it). The exports split by normalised prompt text, the same for every speaker, so a one-kind training set stays disjoint from test and no hard-linked kind export is needed. `python scripts/asr/train.py --check --kinds word --export <export>` prints the remaining counts, and the tests run it on a small export. Training refuses an empty dev set.

Clips are few. Lab 36's 393 single-word and 120 sentence clips include the 166 test clips, so train and dev hold about 271 single-word and 76 sentence clips over three speakers, roughly 90 and 25 each (M02 has the fewest sentences). At batch 2 with 8 accumulations that's about 5 updates an epoch for the word adapter, 2 for the sentence adapter and 7 for the control, so roughly 75, 30 and 105 over 15 epochs. The 10-update warmup is a third of the sentence adapter's run. These are estimates; the stage prints real counts before training.

Dev only has to be non-empty, since `--select last` picks nothing from it. If a speaker lacks a needed kind in train or dev the stage stops before training, and `routing_arms=word` drops the sentence adapter. Lab K ran both kind-only recipes on all speakers and seeds with no failure. `routing_kind_epochs` changes the word and sentence adapters' epochs together. 45 would give the word adapter about twice and the sentence adapter about four times the control's updates per clip. Those runs get an `-e45` suffix and the control stays at 15.

Numbers from earlier runs, for comparison (wrong words, three-seed means):

| System | All 425 | Sentences (of 303) | Single words (of 122) |
| --- | ---: | ---: | ---: |
| Stage P, adapter at 6e-4 on the pretrained model | 178.0 | 85.7 | 92.3 |
| Lab 34's block 8 on the rebuilt base | 174.3 | 83.3 | 91.0 |
| Pretrained model alone (lab 34, seed 17) | 244 | 141 | 103 |
| Whole-block from pretrained, 8e-5, 20 epochs | 192.3 | 107.3 | 85.0 |
| Each of those two on its own kind | about 170.7 | | |
| Adapter on stock, both kinds (lab K) | 212.0 | 117.3 | 95 |
| Adapter on stock, single words only (lab K) | 280.7 | 180 | 101 |
| Adapter on stock, sentences only (lab K) | 246.0 | 117 | 129 |

Lab K, the closest earlier test, points the other way. On the stock base a single-word-only adapter made single words worse (101 against 95) and sentences much worse (it never saw one). A sentence-only adapter left sentences unchanged and made single words worse (129, since inserted words count). Lab K concluded the kinds support each other. This test has a different base (one that already heard the other severe speakers' words) and only uses the word adapter where a detector says single word, but it should still be read against lab K.

Some arithmetic. The best single-word result so far is 85.0, 7 fewer than stage P's adapter, and even a perfect detector with that model gives about 170.7. The bar needs about four times that gain.

The detector's accuracy comes from lab 36's pools (held out by cross-validation, plus test clips) and isn't in a committed table yet. Here it's measured again on the test clips per speaker and seed (`routing/*.json`), each direction counted separately: a sentence sent to the word adapter loses nearly all its words, a missed single word just keeps the control's answer.

## Plan

1. Rebuild stage P's pretrained models if their weights were cleaned.
2. Seed by seed (17 for all speakers, then 18, then 19, so a stopped run leaves a whole seed), train and decode the control, word adapter and sentence adapter. 27 trainings.
3. Release each base once everything that needs it is saved.
4. Compute routed decodes (words, duration and oracle detectors, each with the control and the sentence adapter, plus the whole-block reference when present) and score each with `evaluate.py --hyp` by kind.
5. Write `table.json` (`pool_table.py`, pooled with intervals, then sentences only), `table-words.json` (`pool_table.py --kind word`), `kind-table.json` (`kind_route_report.py`: means by kind, each system against the control seed by seed, the rules, detector mistakes) and `runs.json` (per-run record, adapter sizes).

Decision rules, set before running. The main comparison is "first decode has one word: word adapter + control" against this test's control, three-seed means, same seeds and base.

Accuracy uses the project's rule: at least 2 of 3 speakers move the same way and the pooled count moves by more than about 4.5 points of WER (19 words) on 425. The bar is 148 wrong or fewer.

Single words (the hypothesis): a clear gain is at least 10 fewer wrong of 122, at least 2 of 3 speakers agreeing, and the paired 95% interval over clips above zero at every seed. 10 is above the roughly 8 words two three-seed means need to be told apart (docs/measurements.md, the cross-experiment comparison) and a third of the 30 needed. `table-words.json` decides it.

Sentences are harmed if they get more than 5 words worse than the control on 303. That can only happen by misrouting sentence clips, and harm fails the hypothesis whatever single words did. The sentence adapter helps only if it gets at least 10 fewer wrong on sentences, with at least 2 of 3 speakers.

For the device only word count and duration count; the oracle is a bound. A passing word adapter also has to be a small file (size in `runs.json`), and the second decode's cost gets measured later on the Pi. Thresholds weren't tuned: word count is exactly one word, and the duration cut is the most accurate on the speaker's own train and dev clips (shorter on a tie).

What the outcomes would mean, written before the run. If the word adapter passes with the word-count detector, a device kind split works without a whole-block file, and next comes int8 export of both adapters, the cost of two decodes per single-word clip, and the Pi. If the pooled count also clears 148 the bar is met. If not, the single-word gap remains, now with a measured floor. If the oracle passes but word count doesn't, the detector is the limit, and the routing JSONs list misrouted clips by direction. If even the oracle fails on single words, a word adapter doesn't beat the control even on its own clips, like lab K. Then the kind split with adapters is closed, and the whole-block rows say whether a bigger word model still helps.

If the sentence adapter helps sentences, a two-adapter device would have a use for it. If not, the control's sentence result already comes from sentence clips. A word adapter that's worse than the control on single words by itself would mean one-kind training loses more than it gains on this base. And if whole-block beats the word adapter under routing, the file size is buying accuracy, and the comparison says how much.

## Why

If the device can pick the right model per clip, a second small adapter could close part of the single-word gap without a second 2.4 GB model. If not, one of the cheapest remaining ideas is closed and I'll know whether the word adapter or the detector is the limit.

## Limits

This shows whether a single-word adapter on stage P's base beats the ordinary adapter on single words, whether a device-side detector routes clips correctly, what that does to sentences, and how a perfect detector compares.

It doesn't tune learning rate or epochs for the kind adapters (one recipe, plus one `routing_kind_epochs` point), and doesn't cover base-to-base variation (one base seed), two-decode latency or memory on the Pi, the int8 export, or the intended user. TORGO's single words are isolated with silence around them, so the detector probably does better here than in real use. The test clips have been scored in many earlier experiments, so this is exploratory, with three-seed means and the interval rule as the guard.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

Not yet written.

### Commands and Setup

Times are estimates from lab 34, where 36 adapters with decodes took 100 minutes (about 2.8 minutes each, including battery decodes this test skips).

On Yack or the lab computer, from `app/`, with nothing else on the GPU:

```bash
python lab.py queue routing --dry          # the plan, nothing runs
python lab.py run 37                       # about 1.5 to 2 hours; about 25 minutes more if the pretrained models are rebuilt
python lab.py --set push_results=on queue routing --detach    # in the background, results saved on a lab-results-* branch
```

27 trainings: 9 control (full clip set, about 3 minutes each with decode), 9 word (about three quarters of the clips, shorter ones, about 2.5 minutes) and 9 sentence (about a quarter of the clips, about 1.5 minutes, mostly loading and decoding). Then about 54 routed decodes (72 with the whole-block reference), each scored with `evaluate.py --hyp` in seconds to a minute.

`--set routing_arms=word` skips the sentence adapter (18 trainings, about an hour). `--set routing_kind_epochs=45` gives the sentence adapter about the control's update count (and the word adapter more), as separate `-e45` runs. If training runs out of GPU memory, put `--set xflags=--offload-optimizer` before `run`.

Disk: about 7 GB for the three pretrained models plus one small adapter at a time. On a computer with under 100 GB free, each run's weights are removed after its decode (`trim_models`).

Settings: `routing_arms`, `routing_kind_epochs`, `trim_models`, `xflags`.

Kind filter by hand: `python scripts/asr/train.py --check --kinds word --export data/public/torgo/export/M01 --out /tmp/x` prints the remaining train and dev clips. Routed decode by hand: `python scripts/asr/kind_route.py --manifest data/public/torgo/export/M01/test.jsonl --detector words --word-hyp reports/finetune-torgo/lab-37-word-M01-s17-test-nemo --other-hyp reports/finetune-torgo/lab-37-all-M01-s17-test-nemo --out <new>.jsonl --stats <new>.json`.

Outputs: `reports/lab-37-kind-routing/table.json`, `table-words.json`, `kind-table.json`, `runs.json`, `routing/<detector>-<other>-<speaker>-s<seed>.json` (confusion counts and clips sent to the word adapter), `finished.json`. Decodes and scores are in `reports/finetune-torgo/lab-37-*`. `python scripts/asr/sentence_rescore.py` on the lab computer scores them all on sentences only (families `lab-37-*`).

### Full Results

Not run.

### Conclusions

Not run.

</details>

## Related

[Lab test 34](../lab-34-pretrained-adapter-positions/notes.md), [Lab test 36](../lab-36-lm-rescoring/notes.md), [Lab test 28](../lab-28-deployable-export/notes.md), [Stage P and lab K](../f2-adapter-levers/notes.md)
