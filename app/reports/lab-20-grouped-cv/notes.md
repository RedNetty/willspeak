# Lab test 20: grouped cross-validation, choosing a recipe on every labelled clip instead of 112 dev words

Proposal: evaluation method, not a proposal item. Prototype bar: 35%.

Status: run 16 October 2026 (`1016-retry`, lab computer). No prediction was written first. See Results.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

The severe speakers' dev sets are a few prompts each, 112 words pooled over M01, M02 and M04. The long-adapter study picked stock-form targets at 200 epochs on 9 of those words, and it didn't hold on test. Here I choose on every train and dev clip instead, each decoded by a model that never saw its prompt.

## Setup

Each speaker's clips are split into folds (`--cv-folds`, default 4) by whole prompt, so no prompt is on both sides of a cut. Each recipe trains once per fold on the other folds and decodes the held-out one.

The recipes (`cv_arms`) are `adf1`, the adapter on F-1 and the control as the best system so far, `ad`, the adapter on stock, `adf1e30`, the control at 30 epochs, and `adf1b4`, the control after block 4. Training uses `--select last`, so the held-out fold (also the folder's dev split) never picks an epoch. F-1 never saw the speaker either. The test clips aren't touched until the winner is trained on all his labelled clips and decoded once.

## Plan

1. `cv_folds.py make` writes `cv-<speaker>-all` (decode only) and `cv-<speaker>-f0..f3` as hard links of his audio, and `check` confirms the cuts.
2. Train and decode every recipe on every fold (48 trainings by default), reusing finished folds on a rerun.
3. `merge` joins each recipe's held-out decodes into out-of-fold answers (`oof/`) for lab test 23.
4. `select` ranks recipes by pooled held-out wrong words, with a bootstrap over prompts.
5. If something beats the control, train it on all his labelled clips, decode test once, and pool against stock and the adapter on F-1.

Rule, set before the run: a recipe replaces the control only if it wins by at least `cv_min_gain` (1 point of WER), its paired 95% interval is below zero, and it's better for most speakers. The test comparison then uses the usual 4.5-point rule.

If the control stays, I can stop tuning the adapter's schedule and depth. If a new recipe wins and also beats the control on test, the method is worth reusing. If it wins but test doesn't move, the folds are too small to rank recipes this close. If held-out errors are far above test, the labelled prompts are harder or the folds drop too much vocabulary, so check per-speaker rows first.

## Why

Every recipe choice so far rested on a dev set too small to separate recipes a few words apart, and a bad choice can send the wrong recipe to the prototype.

## Limits

This only ranks these four recipes. It says nothing about recipes I didn't try or about a new speaker. Each fold trains on three quarters of his clips, so every model here sees less data than the final one.

## Results

This ran in queue `1016-retry` on the RTX 4000 Ada (code a9e7242 or later), and I read it from `logs/lab-1016-retry-20.txt`. The stage's `selection.json` is on the lab computer. There are 879 held-out words over M01 (76 clips), M02 (180) and M04 (91).

| Recipe | Wrong of 879 | Rate | 95% interval | Against the control |
| --- | --- | --- | --- | --- |
| `adf1`: adapter on F-1, 15 epochs (control) | 265 | 30.1% | 26.2 to 34.7% | |
| `ad`: adapter on stock | 337 | 38.3% | 33.8 to 43.1% | +8.2 points worse (+4.8 to +11.7), better for 0 of 3 speakers |
| `adf1e30`: adapter on F-1, 30 epochs | 253 | 28.8% | 24.9 to 33.0% | −1.4 (−4.3 to +1.5), better for 1 of 3 |
| `adf1b4`: adapter after block 4 on F-1 | 292 | 33.2% | 28.6 to 38.1% | +3.1 (−0.2 to +6.4), better for 0 of 3 |

The control stays. Thirty epochs was best but not clearly, so nothing new was trained and the test table only has existing systems: stock 323, F-1 248 and adapter on F-1 197 of 425. On held-out prompts the F-1 base beats the stock base with an interval clear of zero. On the 425 test words the same comparison (197 against 208) had been inconclusive. The `oof/` answers went into lab test 23.

The first try on 15 October failed. All 48 fold runs died at once with `ValueError: Clip M01_Session1_0007 audio must be under dev/`, because the folds put held-out audio under `heldout/` and `train.py` wants dev audio under `dev/`. `scripts/asr/cv_folds.py` now writes it under `dev/` (and `test/`), and old folds rebuild automatically since they're just links. To rerun, use `python lab.py run 20`.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

No prediction was written before this run (16 October 2026, queue `1016-retry`), and none is added afterwards.

### Commands and Setup

Run from `app/` in the WSL tree on Yack after `git pull`, with the GPU free and the PC set not to sleep.

```bash
python lab.py run 20
```

At the end the script writes `logs/lab-<stamp>-summaries.txt`, which is the file these notes record. The stage's full output is `logs/lab-<stamp>-<stage>.txt`.

The commands come from `lab.py`. Its flags were checked against each script's options with a dry run (`python lab.py run --dry`, in `tests/unit/test_lab_cli.py` and `tests/unit/test_lab_hardening.py`), and each script's logic has unit tests on synthetic data. Those tests aren't recognition results. At setup on 14 October 2026 no stage had run on the real models or data, and there was no GPU or TORGO audio available to check against.

### Full Results

See Results. Held-out words per fold: M01 [52, 52, 54, 53], M02 [110, 111, 110, 111], M04 [56, 56, 58, 56].

### Conclusions

The F-1 base is the better starting point for the adapter on these speakers' labelled clips. Thirty epochs and block 4 don't change the recipe.

</details>

## Related

[Lab test 23](../lab-23-hypothesis-selector/notes.md), [Adapter on F-1](../f2-m01-m04-adapter8-on-f1/notes.md), [Long adapter](../f2-long-adapter/notes.md)
