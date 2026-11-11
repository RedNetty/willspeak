# Lab test 23: a selector over the systems' answers for each clip

Proposal: decoding (combining systems), following lab test 3. Prototype bar: 35%.

Status: run 16 October 2026 (`1016-retry`, lab computer), with no prediction written first. See also the [Report Context](../Report%20Context.md) and the [Measurement guide](../../../docs/measurements.md).

## Idea

Lab test 3 measured two ceilings over answers that already exist, an oracle that always picks the best system's answer and a vote. A device can't run the oracle. Here I build selectors a device could run and measure them on clips they never learned from.

## Setup

The systems are stock, F-1, the adapter on stock and the adapter on F-1. The methods:

- `vote`, a majority per word
- `wvote`, a vote weighted by each system's accuracy on labelled clips
- `medoid`, the answer closest to the others
- `learned`, a ridge regression that predicts each candidate's wrong-word share from things a device can compute: which system said it, how many systems agree, its similarity to the others, its length against the clip's duration, and the distance to the nearest phrase on a typed list

The selector only learns from out-of-fold answers on his labelled clips. Stock and F-1 never saw them, and the adapter answers come from lab test 20's held-out folds. Its score there is leave-prompt-out. The phrase list is the other speakers' prompts, and the script refuses one containing a test prompt. Test clips are read once, to count.

## Plan

1. Decode his labelled clips with stock and F-1 (`cv-<speaker>-all`).
2. Take the adapter systems' out-of-fold answers from lab test 20.
3. Fit and score the selectors, write each method's test answers, and pool them against the single systems.

Decision rule, set before the run: the 4.5-point noise rule on the pooled test count against the best single system. The learned selector also has to beat that system on the labelled clips (leave-prompt-out), because 425 test words alone are too few.

If learned beats every single system on both sets, the systems fail in different places and a device can tell which to trust, so ship it. Better on labelled clips but not at test means it fitted the labelled prompts, so don't ship. If vote or medoid matches learned, use the simpler one. If nothing beats the adapter on F-1 by more than the noise, the systems fail on the same clips and lab test 3's oracle gap is out of reach with these features.

## Why

Four models may not fit on a Pi, so any gain here has a cost, and I want to know its size first.

## Limits

This shows the pooled wrong-word count of selectors that only use what a device knows, on held-out clips. It can't show the latency and memory of running several models on the device, or how a fifth system would change things.

## Results

Lab run, 16 October 2026, queue `1016-retry` on the RTX 4000 Ada (code a9e7242 or later), read from `logs/lab-1016-retry-23.txt` and `reports/lab-23-hypothesis-selector/table.json`. The selectors were fitted and scored leave-prompt-out on the speakers' labelled clips (879 words), then applied once to the test clips (425 words).

| Method | Labelled clips, wrong of 879 | Test clips, wrong of 425 |
| --- | --- | --- |
| Stock | 565 (64.3%) | 323 (76.0%) |
| F-1 | 423 (48.1%) | 248 (58.4%) |
| Adapter on F-1 (best single system) | 265 (30.1%) | 197 (46.4%) |
| Adapter on stock | 337 (38.3%) | 208 (48.9%) |
| Vote | 276 (31.4%) | 196 (46.1%) |
| Weighted vote | 280 (31.9%) | 198 (46.6%) |
| Medoid | 303 (34.5%) | 202 (47.5%) |
| Learned | 220 (25.0%) | 205 (48.2%) |
| Oracle | 203 (23.1%) | 159 (37.4%) |

The learned selector is 45 words better than the best single system on the labelled clips and 8 words worse on the test clips. That's the "better on labelled clips but not at test" case: it fitted the labelled prompts, so don't ship it. Vote (196) and weighted vote (198) are level with the adapter on F-1. The oracle is 159, above the 148 bar. The systems fail on the same clips.

An earlier attempt on 15 October failed because the stage needed lab test 20's out-of-fold answers (`reports/lab-20-grouped-cv/oof/adf1-M01.jsonl`), and those didn't exist since lab test 20 had failed. Run lab test 20 first (it's fixed now, see its note), then this stage.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

No prediction was written before this run (16 October 2026, queue `1016-retry`), and none is added after the fact.

### Commands and Setup

Run from `app/` in the WSL tree on Yack after `git pull`, with the GPU free and the PC set not to sleep.

```bash
python lab.py run 23
```

At the end the script writes `logs/lab-<stamp>-summaries.txt`, which is the file recorded in these notes. The stage's full output is `logs/lab-<stamp>-<stage>.txt`.

The commands come from `lab.py`. Its flags were checked against each script's own options with a dry run (`python lab.py run --dry`, see `tests/unit/test_lab_cli.py` and `tests/unit/test_lab_hardening.py`), and each script's logic has unit tests on synthetic data. Those tests aren't recognition results. At setup on 14 October 2026 no stage had been run on the real models or data, and there was no GPU or TORGO audio available to validate with.

### Full Results

See Results. `results.json` has the per-method counts on both sets, and `table.json` the pooled test rows with intervals.

### Conclusions

No selector over these four systems beats the adapter on F-1 at test. The gap to the oracle can't be reached with features a device can compute, so selection is closed as a route to the bar.

</details>

## Related

- [Lab test 3](../lab-03-hypothesis-bounds/notes.md)
- [Lab test 20](../lab-20-grouped-cv/notes.md)
