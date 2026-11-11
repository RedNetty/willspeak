# Lab test 22: test-time adaptation on his own recordings, no transcripts

Proposal: device behaviour (personal adaptation as he uses it). Prototype bar: 35%.

Status: run 16 October 2026 (`1016-retry`, lab computer). No prediction was written first.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

The device hears him all day and is never told what he meant. Lab test 12 used pseudo labels from phrase snapping. Here the system learns from its own answers instead: test clips arrive in chunks, and after each one the model retrains on his labelled clips plus the answers it gave.

## Setup

The test clips are split into `tta_chunks` (default 4) chunks, mixed by hash of clip id (`tta_order` = hash) or as runs of sorted ids (`id`). The base system, the adapter on F-1, decodes chunk 0. Round k retrains the same recipe from the same start on his labelled clips plus pseudo labels from chunks 0 to k-1, then decodes chunk k.

A pseudo label is kept only if it isn't empty and F-1 and the adapter on F-1 agree with it by at least `tta_agree_min` (0.5). Pseudo clips are capped at `tta_max_pseudo` (1.0) times his labelled clips, and answers equal to a dev prompt are dropped. The script never reads the manifest's text column, only id, audio path, duration and speaker (`tests/unit/test_tta.py` checks a marker word there never reaches training). `tta.py score` gives wrong words per chunk.

Per-clip entropy adaptation would need the RNN-T's internal scores, so it isn't built.

## Plan

1. Per speaker and round: chunk manifest, training export, train, decode.
2. Merge and score the answers, plus both systems per chunk.
3. Pool against stock, F-1 and the adapter on F-1.

Decision rule, set before the run: the 4.5-point rule on the pooled count, and the gain has to hold or grow in later chunks.

A gain beyond noise that grows in later chunks would mean the device should queue its own answers and retrain overnight. No change would mean its labels are only right where it was already right. Worse later would mean it amplifies its own errors (gate too loose or cap too high). A chunk-1-only gain would be a one-off from extra clips of the same prompts, no loop needed.

## Why

With a month to go before the prototype, I wanted to know whether the device can get better after it ships. This is the simplest test of that.

## Limits

This shows whether self-training helps the adapter on F-1 on clips that arrive after training, without transcripts. It doesn't tell us about a longer deployment, a real day's order of clips, or entropy-based adaptation. Each chunk is a quarter of a one-session test set.

## Results

Queue `1016-retry` on the RTX 4000 Ada (code a9e7242 or later), read from `logs/lab-1016-retry-22.txt`, `reports/lab-22-test-time-adaptation/table.json` and the by-chunk tables. Four hashed chunks, three rounds, gate 0.5, seed 17. Pooled wrong words of 425:

| System | M01 | M02 | M04 | Pooled | WER | 95% CI | Empty of 166 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Stock | 131 | 62 | 130 | 323 | 76.0% | 66.7 to 86.3 | 34 |
| Adapter on F-1 | 78 | 47 | 72 | 197 | 46.4% | 40.2 to 53.2 | 3 |
| Adapter on F-1 + TTA | 73 | 41 | 65 | **179** | **42.1%** | 36.3 to 48.9 | 3 |

By chunk (wrong / words; chunk 0 is identical by construction):

| | Chunk 0 | Chunk 1 | Chunk 2 | Chunk 3 |
| --- | --- | --- | --- | --- |
| M01 base → TTA | 16/52 | 20/45 → 19/45 | 17/28 → 11/28 | 25/44 → 27/44 |
| M02 base → TTA | 7/10 | 4/19 → 5/19 | 16/27 → 15/27 | 20/39 → 14/39 |
| M04 base → TTA | 18/53 | 20/32 → 20/32 | 16/32 → 13/32 | 18/44 → 14/44 |

All three speakers improved, by 5, 6 and 7 words. Pooled, the gain is 18 words or 4.2 points, just under the rule. Chunk 0 can't change, though, and on chunks 1 to 3 (310 words) it's 156 wrong down to 138, or 5.8 points. Chunk 1 is flat (44 → 44), chunk 2 better (49 → 39), chunk 3 better (63 → 55), so the gain holds or grows.

The gate let in 17, 29 and 41 pseudo clips for M01 over the rounds (10, 16, 23 for M02; 14, 23, 36 for M04) and rejected 6 for disagreement and 8 for empty answers in total. One seed, but it's the new lowest raw count, level with stage P's pretrained adapter (178.0 over three seeds). The two weren't combined here; that's lab test 25.

An earlier attempt on 15 October failed in round 1 for every speaker with `Clip tta-M01_Session2_3_0026 audio must be under train/`, since the pseudo clips were written under `pseudo/`. Its summary table only had stock, F-1 and adapter-on-F-1 rows from other runs. Fixed in `scripts/asr/tta.py` (pseudo audio now goes under `train/`), and `lab.py` rebuilds old-layout round exports. That stage still wrote `table.json`, so rerun with `python lab.py run 22 --rerun` or move the file aside.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

No prediction was written before this run (16 October 2026, queue `1016-retry`), and none is added after the fact.

### Commands and Setup

Run from `app/` in the WSL tree on Yack after `git pull`, with the GPU free and the PC set not to sleep.

```bash
python lab.py run 22
```

At the end the script writes `logs/lab-<stamp>-summaries.txt`, which is the file these notes are based on. The stage's full output is `logs/lab-<stamp>-<stage>.txt`.

The commands come from `lab.py`. Its flags were checked with a dry run (`python lab.py run --dry`; `tests/unit/test_lab_cli.py`, `tests/unit/test_lab_hardening.py`), and each script has unit tests on synthetic data, which aren't recognition results. At setup on 14 October 2026 no stage had run on the real models or data, and there was no GPU or TORGO audio to check with.

### Full Results

See Results. `table.json` has the pooled rows with intervals and `by-chunk-<speaker>.json` the chunk tables.

### Conclusions

Self-training on its own agreed answers helped every speaker, and helped more in later chunks, at one seed. That's the "gain growing in later chunks" outcome. It lands just under the pooled rule because a quarter of the clips can't change. Lab test 25 repeats it at three seeds on the pretrained adapter.

</details>

## Related

[Lab test 12](../lab-12-pseudo-label-loop/notes.md), [Adapter on F-1](../f2-m01-m04-adapter8-on-f1/notes.md)
