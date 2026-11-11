# Lab test 36: can a small text model pick the right sentence from Parakeet's own guesses?

Proposal: F-2 (extension) and H4 (the model has to run on hardware he can carry). The measure is whole sentences right, for any sentence, with sentence WER beside it.

Status: written and run 6 November 2026 on the lab computer (queue `1106-1346`, code `258f1205`, 13:47 to 19:17 lab clock, 27 language models, no failures). Results are on `origin/lab-results-1106-191759-075465`, not merged to master, and were written up 9 November 2026. No prediction was recorded before the run. An earlier queue (`1106-1234`) ran out of disk while training the pretrained models (fixed in `5d8750bc`). Nothing measured on the Pi yet.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

Parakeet often has the right sentence among its guesses and still outputs a wrong one. Stage P's adapter turned "The little schoolhouse stood empty." into "the level of graham stood empty", which sounds close but isn't English anyone would say. A text language model never hears the audio but knows what English looks like, so here it gets to choose among Parakeet's guesses.

Parakeet decodes each clip several ways (the views below) and every distinct answer goes in that clip's pool. The recogniser scores how well each answer fits the sound (its likelihood summed over every alignment) and the language model scores how likely it is as English. The kept answer has the highest `total = audio score + weight × language score + bonus × number of words`. The bonus is there because both models find shorter answers cheaper, since every extra word is one more chance to be unlikely.

The language model can only pick what Parakeet proposed, so the share of clips whose pool holds the right sentence is the ceiling.

## Setup

Two recognisers. Stock is Parakeet-TDT-0.6B-v2, never trained on TORGO. Stage P's adapter is the expo recipe: a model trained on the other severe speakers, then an adapter after block 8 at 6e-4 on the target speaker (seed 17, trained fresh as in lab test 28), decoded in NeMo at full precision and not as the Pi's int8 file. Each table is scored separately:

| Table | Recogniser | Clips | Why |
| --- | --- | --- | --- |
| Every severe sentence | stock | 292 sentence clips: F01 20, M01 90, M02 93, M04 89 (the T-1 manifest) | The most clips. Stock never saw TORGO, so every clip is fair. |
| The test clips | stock and stage P's adapter | 166 clips of M01, M02 and M04: 44 sentences (18, 10, 16) and 122 single words | The expo models themselves, on the set every lab table uses. |
| Every labelled clip, held out | stock and stage P's recipe | 513 clips of M01, M02 and M04: 120 sentences and 393 single words | The 44 test sentences are too few to settle anything. Here the train and dev clips are added by cross-validation: four folds that hold out whole prompts, the adapter retrained without each fold, the held-out fold decoded. With the test clips, every labelled clip is decoded by a model that never trained on its prompt. The pretraining never saw the speaker at all. |

The fold adapters train on three quarters of the train and dev clips, so they're the recipe, not the expo files.

### Views

`lm_views` gives nine decodes per clip: `greedy` (the normal decode, always first, so "first decode" is what the device says today), `beam4` and `beam8` (every answer of a 4- and 8-wide beam), `tempo1.1`, `tempo1.2` and `tempo1.3` (10, 20 and 30% faster at the same pitch, since severe speech is slow and the model learned ordinary rates), and `pitch0.9`, `pitch0.95` and `pitch1.05` (pitch and formants down 10%, down 5% and up 5% at the same speed). Answers differing only in capitals or punctuation count as one, and all are scored against the unchanged audio. Answers more than 15 nats behind the clip's best audio score (a factor of about three million) are dropped, except the first decode.

### What the language model reads

Every answer in one form: lower-case words, capital first letter, closing full stop ("The boy ran to the store.").

The first word needs something before it, and models differ: some want their start token, some read better after a new line than their end-of-text token. The script tries all three on three fixed non-TORGO sentences (first: "The sun rises in the east and sets in the west."), keeps the simplest that works, and saves it in the score file. A model worse than -5 per token on them is refused (working models sit near -2 to -3, random near -12), so a misread file can't add noise.

### Weights

Weight and bonus come from a fixed grid (weights 0 to 3, bonuses 0 to 8), chosen on the other speakers' sentence clips, so no speaker picks his own. Ties go to the smaller weight, then smaller bonus. The "audio score only" row is weight 0, re-ranking with no language model. The last column applies one preset pair (0.7 and 3) to everyone.

### Language models

Listed by `python scripts/asr/lm_score.py list`. All are free downloads, none trained here. All but the first run through llama.cpp from 4-bit or 8-bit files, a form a Pi could hold. Base models where a base file exists, chat-tuned ones marked. `lm_models` takes group names or tags. `pi` is the default group, each file under 1.7 GB.

| Tag | Model | Released | Parameters | File |
| --- | --- | --- | ---: | ---: |
| `qwen25-05b-fp32` | Qwen2.5-0.5B, full precision, through torch | Sep 2024 | 0.49B | 988 MB |
| `qwen25-05b-q8`, `-q4` | Qwen2.5-0.5B, 8-bit and 4-bit | Sep 2024 | 0.49B | 531, 398 MB |
| `smollm2-360m-q8` | SmolLM2-360M | Nov 2024 | 0.36B | 386 MB |
| `gemma3-270m-q8` | Gemma 3 270M | Aug 2025 | 0.27B | 292 MB |
| `granite4-350m-q4` | Granite 4.0 350M | Oct 2025 | 0.35B | 237 MB |
| `qwen3-06b-q4` | Qwen3-0.6B | Apr 2025 | 0.60B | 397 MB |
| `lfm25-230m-q4` | LFM2.5-230M (chat-tuned) | Jun 2026 | 0.23B | 153 MB |
| `lfm25-350m-q4` | LFM2.5-350M (chat-tuned) | Mar 2026 | 0.35B | 229 MB |
| `qwen35-08b-q8`, `-q4` | Qwen3.5-0.8B | Mar 2026 | 0.75B | 812, 529 MB |
| `minicpm5-1b-q4` | MiniCPM5-1B | May 2026 | 1.08B | 688 MB |
| `lfm25-12b-q4` | LFM2.5-1.2B | Jan 2026 | 1.17B | 731 MB |
| `gemma3-1b-q4` | Gemma 3 1B | Mar 2025 | 1.00B | 806 MB |
| `llama32-1b-q4` | Llama 3.2 1B | Sep 2024 | 1.24B | 808 MB |
| `granite4-1b-q4` | Granite 4.0 1B | Oct 2025 | 1.63B | 1,024 MB |
| `qwen25-15b-q4` | Qwen2.5-1.5B | Sep 2024 | 1.54B | 986 MB |
| `qwen35-2b-q4` | Qwen3.5-2B | Mar 2026 | 1.88B | 1,274 MB |
| `qwen38-2b-distill-q4` | Qwen3.8-2B distill (chat-tuned; made by Empero, not Alibaba) | Aug 2026 | 1.94B | 1,312 MB |
| `spark26-17b-q4` | Spark-X2.6-1.7B | Oct 2026 | 1.71B | 1,108 MB |
| `minicpm5-2b-q4` | MiniCPM5-2B | Sep 2026 | 2.52B | 1,561 MB |
| `lfm25-26b-q4` | LFM2.5-2.6B | Jul 2026 | 2.70B | 1,675 MB |

`large` has 2 to 3 GB files that fit beside Parakeet on an 8 GB Pi but would be slow. Add them with `--set lm_models="pi large"`.

| Tag | Model | Released | Parameters | File |
| --- | --- | --- | ---: | ---: |
| `granite42-3b-q4` | Granite 4.2 3B (chat-tuned) | Aug 2026 | 3.66B | 2,244 MB |
| `nanbeige42-3b-q4` | Nanbeige4.2-3B | Jul 2026 | 4.17B | 2,575 MB |
| `spark26-4b-q4` | Spark-X2.6-4B | Oct 2026 | 4.11B | 2,600 MB |
| `qwen35-4b-q4` | Qwen3.5-4B | Mar 2026 | 4.21B | 2,709 MB |
| `gemma4-e2b-q4` | Gemma 4 E2B (chat-tuned) | Mar 2026 | 4.65B | 3,107 MB |

`ceiling` isn't for the Pi: `qwen38-27b-q4`, Alibaba's Qwen3.8-27B (Aug 2026, 16.5 GB file, about 18 GB of memory), to see whether a far stronger text model chooses better. It's too big for the machine the others were checked on, so its first load is the run itself. Qwen3.8 (August 2026) only comes at 27B and up. The Pi-sized one with that name is Empero's Qwen3.5-2B trained to copy Qwen3.8.

Left out: LFM3-1.7B (Oct 2026) and K2-Horizon-0.9B (Sep 2026), which the newest llama-cpp-python (0.3.36, 1 October 2026) can't load and which at full precision need their own repository's code. `smollm2-135m-q8`, `smollm2-17b-q4` and `qwen3-17b-q4` are in the catalog but in no group.

### Memorisation check

TORGO's prompts are public, so a model might know them by heart. Each model gets the first half of every distinct test sentence of four words or more, and the table counts how many it finishes word for word.

### Earlier numbers for comparison

Stock on the 292 severe sentences got 59.59% WER as int8 through sherpa-onnx (T-1). No system has had whole sentences right counted before. Stage P's adapter on the test clips gets about 176 to 183 wrong of 425 and 28.3% on the 303 sentence words, with no empty answers in NeMo (lab tests 28 and 34).

Lab test 6 found no clip where the true words beat the wrong answer under the model's own likelihood (with its normalisation caveat), so "audio score only" shouldn't gain much. Lab test 23's learned selector over several systems did worse than the best single system, but that used other systems' answers, not one recogniser's own guesses with two numbers set on other speakers.

An earlier CPU pilot (not a project result, different pools, clips and code) ran stock on 266 severe sentence clips. Qwen2.5-0.5B at full precision moved whole sentences right from 9.8% to 13.5% (14 gained, 4 lost), GPT-2 gave no clear gain, and the pool held the right sentence for 19.5%. On one speaker the fine-tuning and language-model gains roughly added. The views here were chosen after a pilot look at pool hits on a third of the stock clips, so the stock pool ceiling isn't a clean estimate, though the weights still come from other speakers. The first-word prefix was set on the three fixed sentences, not on TORGO.

## Plan

1. Language-model environment (`./willspeak.sh lm-setup`, or built by the test if missing).
2. Stock on every severe sentence: pools, scores, table, no training.
3. Stage P: pretrained models and adapters (rebuilt if gone), fold adapters, pools for both systems on the same clips.
4. Each language model in turn: download, score every answer once, memorisation check, delete the file if disk is short. Then one report with the three tables and each model's size, memory and speed.

Decision rules, set before the run. `qwen25-05b-q8` is named in advance, as the model already tried, in a Pi-sized form. It counts as a gain on a table when the 95% interval of the change in sentences right against the first decode is above zero. The rest are a screen, and the best of two dozen looks better than it is, so they count only if above zero on both the stock table (292 clips) and the stage P all-clips table (120 sentence clips). Single words, read from the "single words right" column, must not lose more clips than the sentences gain. On the device a model must fit next to Parakeet's 0.9 to 1.3 GB, with seconds per clip measured on the Pi later.

A gain on both systems means the two stack and the Pi comes next. Stock only means the adapter already fixed what a language model sees, so rescoring is for people without a personal model. No gain with a high pool ceiling means a better chooser is needed. A ceiling near the first decode means wider pools or a better recogniser come first. If 4-bit loses what 8-bit or full precision gains, the Pi form is the problem, not the idea.

## Why

Whole sentences right is what a listener cares about, and the fine-tuning work hasn't measured it. Rescoring needs no training or recordings of him, so it would apply to his personal model the day it exists.

## Limits

No Pi speed or memory (measured there later), no int8 recogniser (pools come from NeMo at full precision), and nothing about the intended user's speech. TORGO sentences are read prompts, many shared across speakers, so the same sentence from two speakers isn't two independent tests. The table also saves an interval that resamples prompts. The test clips have been scored in many earlier experiments.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

None was recorded before the run.

### Commands and Setup

All times below are estimates.

On Yack, run from `app/` in the WSL tree with nothing else on the GPU. The pool builder stops at once if the card has under 3.5 GB free. The test builds the language-model environment (`~/venvs/willspeak-lm`: llama-cpp-python, CPU build) if it's missing, or `./willspeak.sh lm-setup` builds it ahead of time. Each language model downloads into `models/lm/` when its turn comes and stays there, since Yack has the disk.

A first look with stock only, the named model only and no training (about 20 minutes):

```bash
python lab.py --set lm_systems=stock --set lm_models=qwen25-05b-q8 run 36
```

Everything, meaning three pretrained models, fifteen adapters, about 1,300 clip pools and the `pi` group (5 to 7 hours on the 4060, reusing pools and scores already made):

```bash
python lab.py run 36 --rerun
```

`--rerun` moves the earlier `table.json` aside and isn't needed if there's no earlier table. If training runs out of GPU memory, put `--set xflags=--offload-optimizer` before `run`. `--set lm_models="pi large"` adds the five larger models (2 to 3 hours more). `--set lm_threads=16` scores faster on a CPU with the cores (default 8).

On the lab computer it's one command, run from `app/` in the lab checkout once lab test 36 is on the branch checked out there:

```bash
./willspeak.sh lab-lm
```

It pulls the code, runs `lab-setup` if the training environment or TORGO is missing, builds the language-model environment, downloads and checks the named model, and runs `python lab.py check`. Then it starts the queue in the background with `lm_models="pi large"`, `push_results=on` (results saved on a `lab-results-*` branch and pushed) and `wait_busy=on` (waits for any other lab run to finish).

`./willspeak.sh lab-lm --quick` is the stock-only first look with the named model. Running `./willspeak.sh lab-lm` again remakes the table and reuses everything already made. `--models "pi"` leaves out the larger five and `--dry` just prints the plan. By hand, the queue step is `python lab.py --set lm_models="pi large" --set push_results=on queue 36 --detach`.

The queue keeps going after the terminal closes, and `logs/lab-queue-<stamp>.out` shows where it's at. With under 100 GB free, the test deletes each language-model file once its scores are saved and removes each fine-tuned model after its pools. At most it holds the three pretrained models (about 7 GB) and later one language-model file (about 3 GB). Estimated 7 to 10 hours with `pi large`.

Adding `ceiling` to `lm_models` also scores Qwen3.8-27B, which needs 16.5 GB of disk while it's scored, about 18 GB of memory and several more hours. `./willspeak.sh lab-remaining` also reaches lab test 36, after everything else unfinished there, using the `lm_models` saved by `python lab.py set lm_models "pi large"`.

Afterwards on another computer: `git fetch`, then `python lab.py space fetch-results origin/<newest lab-results branch>`.

To check a model file by hand, run `python scripts/asr/lm_score.py fetch --models <tag>`, then `check --models <tag>` (with `~/venvs/willspeak-lm/bin/python`). That prints what goes before the first word and how the model completes a simple sentence. `release --models <tag>` deletes the file.

Settings: `lm_systems`, `lm_models`, `lm_views`, `lm_threads`, `lmpy`, `cv_folds`, `trim_models`.

The tables print at the end of `logs/lab-<stamp>-36.txt` and are saved as `reports/lab-36-lm-rescoring/table.json`. Pools are in `pools/`, each model's scores in `scores/`, and the memorisation check in `completions/`.

### Full Results

Queue `1106-1346`, `lm_models="pi large"` (27 models), all three systems. The tables are in `reports/lab-36-lm-rescoring/table.json` on the results branch, with the pools, scores and memorisation files beside it. Intervals are 95%, paired, 2,000 resamples of clips. The table also has an interval that resamples whole prompts, and the two agree to within about a point. Weights are chosen on the other speakers.

The right sentence is somewhere in the pool for 16.4% of the 292 stock sentences (first decode 9.6%) and 37.5% of stage P's 120 held-out sentences (first decode 24.2%). Re-ranking by audio score alone gains no sentence on either (stage P: 29 of 120 before and after), so every gain below comes from the language model.

Stage P's recipe on the 513 held-out clips (120 sentences, 393 single words). The first decode gets 29/120 sentences right, sentence WER 27.1% (247 of 911 words wrong), and 221/393 single words right.

| Model | File / peak MB / s per clip | Stock, 292 sentences (change in points) | Stage P, 120 sentences | Single words right of 393 (first decode 221) |
| --- | --- | --- | --- | --- |
| `qwen25-05b-q8` (named in advance) | 531 / 666 / 0.39 | +1.0 [-1.0, +3.1] | +8.3 [+3.3, +13.3] (39/120) | 210 |
| `qwen25-05b-q4` | 398 / 570 / 0.42 | +3.1 [+1.0, +5.5] | +7.5 [+2.5, +12.5] (38/120) | 195 |
| `qwen25-05b-fp32` | 988 / 3,676 / 0.90 | +2.4 [0.0, +4.8] | +8.3 [+4.2, +13.3] (39/120) | 208 |
| `spark26-17b-q4` | 1,108 / 1,832 / 0.83 | +3.1 [+0.7, +5.5] | +7.5 [+2.5, +13.3] (38/120) | 211 |
| `gemma3-270m-q8` | 292 / 464 / 0.27 | +1.4 [-0.7, +3.4] | +7.5 [+2.5, +13.3] (38/120) | 207 |
| `llama32-1b-q4` | 808 / 1,391 / 0.60 | +2.4 [-0.3, +5.1] | +9.2 [+4.2, +15.0] (40/120) | 194 |
| `qwen35-4b-q4` (large group) | 2,709 / 4,442 / 2.08 | +0.7 [-1.4, +2.7] | +10.8 [+5.8, +16.7] (42/120) | 194 |
| `gemma4-e2b-q4` (large, chat-tuned) | 3,107 / 4,322 / 1.23 | +0.3 [-1.7, +2.4] | +2.5 [-2.5, +7.5] (32/120) | 201 |

All 27 models are in `table.json`. On stage P, 24 of 27 have an interval above zero. Of the 16 sentences the pool holds but the first decode misses, the median model captures 9 (56%) and the best 13 (81%). On stock the median captures 3 of 20 (15%) and the best 9 (45%). Stage P sentence WER falls from 27.1% to 22.5 to 24.8% for most models (`qwen25-05b-q4` 22.5%).

The stock and stage P rankings barely agree across models (Spearman 0.22). File size doesn't predict the stock ranking (0.03) and only moderately predicts the stage P one (0.50), so picking one model from this table is mostly picking noise. Chat-tuned models are the weakest group on stage P. The two small LFM2.5 models and Gemma 4 E2B gain 2.5 points with the interval including zero, while Granite 4.2 3B and the Qwen3.8 distill are ordinary. The memorisation check is clean: every model finishes only 1 to 3 of the 90 test sentences from their first half.

On the expo test clips (44 sentences, 122 single words, stage P's adapter) the first decode gets 8/44 sentences right (18.2%) with 84 of 303 sentence words wrong. The named model gets 10/44 (+4.5 [0.0, +11.4]). As the plan said, 44 sentences settle nothing.

Against the decision rules:

The named model, `qwen25-05b-q8`, is above zero on stage P (+8.3 points) but not on stock (+1.0, interval -1.0 to +3.1), so it isn't a gain on both tables.

Two models pass the screen (above zero on stock and stage P): `qwen25-05b-q4` and `spark26-17b-q4`. `qwen25-05b-fp32` passes only on the prompt-level interval on stock (+0.3), not the clip interval the rule used.

The single-word rule fails for every model with a stage P gain. The same weights on single-word clips lose 10 to 31 of 393 while the sentences gain 7 to 13 (named model +10 sentences and -11 single words, `qwen25-05b-q4` +9 and -26, `spark26-17b-q4` +9 and -10). Only the two small LFM2.5 models pass (+6 and -2 single words), and their sentence gains aren't above zero. The weights were fitted on sentence clips only, so single words inherit a per-word bonus tuned for sentences.

Device fit wasn't measured on the Pi.

Replays on the saved pools, 9 November (no new run, `lm_report.py --single-word-chooser`). Wrong words over all 1,304 words of the 513 held-out clips, first decode 427 (32.7%):

| | `qwen25-05b-q8` | `qwen25-05b-q4` | `spark26-17b-q4` | `qwen35-4b-q4` |
| --- | --- | --- | --- | --- |
| Every clip rescored | 400 | 411 | 408 | 414 |
| Single words left at the first decode (clips whose first decode has one word) | 390 (29.9%) | 383 (29.4%) | 396 (30.4%) | not run |
| Single words chosen by their own weights (language model, votes, per-word term; fitted on single-word clips of the other speakers) | 387 (29.7%) | 376 (28.8%) | 389 (29.8%) | 377 (28.9%) |

On the 425-word test count (44 sentences, 122 single words, first decode 176 wrong or 41.4%, bar 148), in the same model order: every clip rescored gives 178 / 168 / 169 / 163, single words left alone 171 / 165 / 169 / not run, and the own chooser 171 / 164 / 170 / 161. The 425-word differences are 5 to 15 words, inside the 19-word rule.

The device can tell the kinds apart. "The first decode has one word" calls 384 of 393 single-word clips single and 2 of 120 sentences (117 of 122 and 1 of 44 on the test clips). Clip duration alone overlaps (sentence minimum 2.4 s, single-word 95th percentile 4.1 s).

The single words' own chooser nets +4 to +10 single words of 393 (gains 10 to 12, losses 2 to 6). That's `qwen25-05b-q4` +1.8 points [0.0, +3.8] and `qwen35-4b-q4` +2.5 [+0.8, +4.6]. The audio score plus how many of the nine decodes proposed an answer, with no language model at all, gives +2.3 [+0.5, +4.3] (+9). The fitted language-model weight is 0 to 0.2. I tried a word-frequency term (the `wordfreq` package) in a scratch replay and the fit gave it weight 0. The pool holds the right word for 302 of 393 single-word clips against 221 for the first decode, and the first decode already has the most votes on 363 of the 393. What's missing is a feature that tells the right isolated word from a wrong one.

On stage P the chosen weights agree across speakers (most often 0.7 with a bonus of 6). At least 22 of 81 choices sit on the grid maximum of 8, so the grid is cut off. On stock they scatter, from 0.2 with 1 up to 1.5 with 3. The fixed 0.7 / 3 preset is within a sentence or so of the chosen weights on stage P (mean +0.3 sentences) and slightly better on stock (+2.4).

### Conclusions

Interpretation written 9 November 2026, after the results. No prediction was recorded before the run.

A text model reliably picks better whole sentences from stage P's own guesses. Sentences right go from 24.2% to between 31.7% and 35.0% of 120 held-out sentences (+7.5 to +10.8 points), with 0 to 3 sentences lost, and sentence WER drops from 27.1% to about 23%. It needs no training and no recordings of the user.

The gain gets bigger after fine-tuning. On stock the same models gain 0 to 3 points, on stage P 7 to 11. The pool is what changes. The right sentence is in it for 16.4% of stock clips and 37.5% of stage P's, and the near-misses a fine-tuned model makes ("the level of graham stood empty") are the kind a text model can reject. That's the opposite of the "gain on stock only" outcome, and it means the two gains stack.

As run, it doesn't meet the plan's device rules. The named model fails the stock table, only two models pass the screen, and every model with a gain loses single words under the sentence weights. Routing single-word clips around the sentence weights (detected by a one-word first decode, about 97% accurate) removes the loss. With their own weights they gain a little, +1.0 to +2.5 points of single words right. The interval is above zero for `qwen35-4b-q4` and `spark26-17b-q4` and touches zero for `qwen25-05b-q4`.

It doesn't reach the bar. The 425-word count moves from 176 to between 164 and 171 for the Pi-sized models in the best arrangement here, inside the 19-word rule and at least 16 above 148. Single words, which the language model can't help much, are the whole gap (see the comparison in [measurements.md](../../../docs/measurements.md)).

Which model to use isn't settled. Models within about a point of each other on stage P can't be ranked. The two that pass the screen are `qwen25-05b-q4` (398 MB, 0.42 s per clip on the lab computer) and `spark26-17b-q4` (1.1 GB, 0.83 s, loses fewer single words). Both lost single-word clips under the plain rule.

Still open, in order: the Pi's cost (mostly the nine decodes per clip, not the language model), a wider bonus grid, averaging several models' scores, the pool ceiling (45/120) against the chooser's 38 to 42, which more candidates (other seeds, other adapter positions) could lift, and single words, where the pool holds the answer for 77% of clips and the chooser finds 58%.

</details>

## Related

[Lab test 6](../lab-06-reference-score/notes.md), [Lab test 23](../lab-23-hypothesis-selector/notes.md), [Lab test 28](../lab-28-deployable-export/notes.md), [Lab test 29](../lab-29-hard-core/notes.md), [Stage P](../f2-adapter-levers/notes.md)
