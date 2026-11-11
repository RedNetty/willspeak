# Lab test 7: stock against fine-tuned, does a model's stock WER predict its fine-tuned WER

Proposal: F-2 (H3), an extension about choosing the base model cheaply. Prototype bar: 35%.

Status: run 7 October 2026 on M02 and M04 only (256 reference words). Results read 14 October.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

Lab test 4 ranks models by stock WER in minutes, which only helps if the ranking survives fine-tuning. Here I fine-tune four models the same way on M02 and M04 and compare their stock and fine-tuned order.

## Setup

Pooled means wrong words summed over M01 (169 test words), M02 (95) and M04 (161), 425 in all. Stock Parakeet-TDT-0.6B-v2 gets 323 wrong (76.0%), the best so far is about 201 to 210 (about 48.5%, adapter after block 8, three seeds), and the 35% bar is 148 or fewer.

The models are the project's v2, Parakeet-TDT-0.6B-v3, Parakeet-TDT-1.1B and Parakeet-RNNT-0.6B (`TUNE_TAGS` changes the list). v2 is retrained with the same code so all four are comparable, and since GPU runs aren't bit-for-bit repeatable, that also shows some spread against the old v2 runs. The recipe is F-2 whole-block: blocks 0 to 7 trained in full, five epochs, last weights, batch 2 with 8 accumulation steps, frozen BatchNorm, normalised targets. 1.1B uses `--offload-optimizer`, and a failed run is retried once at batch 1 with 16 accumulation steps under a name ending `-b1`.

`rank_check.py` gives each model's WER before and after, the Spearman correlation between stock and fine-tuned order (pooled and per speaker), and how many pairs the stock order got right. With four models a 1.0 happens by luck once in 24 orders (once in 6 with three), so a perfect order is only encouraging. A reversed or scrambled order is clearer evidence against the screen.

## Plan

1. Decode the stock models on M02 and M04 (reusing lab test 4's decodes).
2. Fine-tune each model on each speaker's F-2 export, then decode and score his test set.
3. Run `rank_check.py` over models that finished on both speakers.

With two speakers and four models the pooled rule doesn't apply. I just report the correlation, its chance by luck, and whether the order held, flipped or neither. If it holds, lab test 4 can be the selection step. If not, stock WER is only a hint and models have to be fine-tuned to compare. If a model ends clearly below v2, carry that one forward whatever its stock rank.

## Why

Training every candidate is expensive and a decode is cheap. This shows how far the cheap way can be trusted.

## Limits

This orders four models on two speakers with one seed each. It says nothing about other speakers or about models `train.py` can't train. The 1.1B model might not fit on the card, and the run would record that and move on. Seed noise is 4 to 5 points per speaker, so models closer than that aren't ordered.

## Results

Four base models, same whole-block recipe, one seed, scored on M02 and M04.

| Model | Stock WER | Fine-tuned WER | Gain (points) | Fine-tuned wrong |
|---|---:|---:|---:|---:|
| v2 | 75.0% | 66.4% | +8.6 | 170/256 |
| 1.1B | 74.6% | 70.3% | +4.3 | 180/256 |
| RNNT 0.6B | 73.8% | 68.8% | +5.1 | 176/256 |
| v3 | 77.0% | 76.2% | +0.8 | 195/256 |

Stock order only loosely predicts fine-tuned order. Pooled Spearman is +0.40 with 4 of 6 pairs right (a random order matches exactly 4.2% of the time), +0.80 on M04 and -0.40 on M02. v2 ends lowest and v3 gains almost nothing, but at one seed no pair is established. Stock score is a weak guide.

Source: rank_check log (`logs/rank_check-20261007-232737-363482.log`), RTX 4000 Ada lab computer, 6-8 Oct 2026, reviewed 14 October 2026.

Sentences only (secondary score, added 21 October 2026). Same decodes, single-word clips left out: 303 of the 425 words over three speakers, so 35% is 106 wrong or fewer. Means over the seeds listed, read from each run's `results.json` with `scripts/asr/sentence_rescore.py`. The full ranking is in `docs/measurements.md` under "Secondary score: sentences only". These runs only cover some speakers, so each is scored on its own sentence words (M02 alone has 68, M02 and M04 have 181):

| Recipe (run name) | Speakers | Seeds | Sentence words wrong | Sentence WER | All-clip WER |
| --- | --- | ---: | ---: | ---: | ---: |
| `lab-07-v2-first-run` | M02 | 1 | 26.0 of 68 | 38.2% | 51.6% |
| `lab-07-v2` | M02 + M04 | 1 | 97.0 of 181 | 53.6% | 66.4% |
| `lab-07-rnnt06` | M02 + M04 | 1 | 99.0 of 181 | 54.7% | 68.8% |
| `lab-07-1p1b` | M02 + M04 | 1 | 106.0 of 181 | 58.6% | 70.3% |
| `lab-07-v3` | M02 + M04 | 1 | 122.0 of 181 | 67.4% | 76.2% |

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

No prediction was written before this run (7 October 2026), and none is added afterwards.

### Commands and Setup

Run from `app/` in the WSL tree on Yack after `git pull`, with the GPU free and the PC set not to sleep. The script stops if a `train.py` run is already going, and each stage stops straight away if one of its outputs already exists.

```bash
python lab.py run 7
```

At the end the script writes `logs/lab-<stamp>-summaries.txt`, which is the file these notes record. The stage's full output is `logs/lab-<stamp>-7.txt`. Stage 7 is long, so start it in the evening.

The commands come from `lab.py`. Its flags were checked against each script's options with a dry run (`python lab.py run --dry`, in `tests/unit/test_lab_cli.py`), and each script's logic has unit tests on synthetic data. Those tests aren't recognition results. Before this run, none of these had been tried on the real models or data: training any base other than v2 through `train.py --base`, the RNNT model through `train.py`, and `rank_check.py` on real decodes.

### Full Results

The Results section above has the full comparison and its sources.

### Conclusions

Written 21 October 2026 from the recorded results above.

On M02 and M04 (256 reference words), the same whole-block recipe at one seed left Parakeet-TDT-0.6B-v2 lowest at 66.4% (170 wrong). RNNT 0.6B got 68.8%, 1.1B 70.3% and v3 76.2%. At one seed none of these pairwise orderings is established.

Stock scores predicted the fine-tuned order only loosely: Spearman +0.40 pooled, +0.80 on M04 and -0.40 on M02. v3 started level with the others (77.0%) and gained only 0.8 points, while v2 gained 8.6.

Along with lab test 4, where no stock model showed a lead over v2, this supports keeping v2 as the base. A base has to be judged after fine-tuning, and its stock score isn't enough.

The spread between recipes on v2 is now much bigger than the spread between bases here. On the pooled three-speaker set the adapter on stock is at 208 wrong and the adapter on the severe-pretrained model at 178.0, against 291 for this whole-block recipe.

Next step: keep v2. Stage B, the adapter on other bases, is what's left to test on this question, and it has no result yet.

</details>

## Related

[Lab test 4](../lab-04-stock-model-screen/notes.md), [F-2: base-model ceiling](../f2-base-model-ceiling/notes.md)
