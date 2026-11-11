# Lab test 30: acoustic phrase scoring on the pretrained adapter

Proposal: Device design (extension), following lab test 24. Prototype bar: 35%, which is 148 wrong of 425 pooled test words.

Status: Written 20 October 2026 and run in the `1020-new` queue on the lab computer. The queue finished at 02:09 on 21 October (lab clock), and I read the results from the printed tables in `logs/` on 21 October 2026. The tables and reports the queue wrote stayed on the lab computer and aren't in the repository.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

Lab test 24 measured what the device says with its phrase list for stock, F-1 and the adapter on F-1, with thresholds chosen on the other speakers. The adapter on F-1 got 26.4% with a full list and 31.8% at 80% coverage. The best raw system is now the pretrained adapter (178 wrong against 197). But the list closed most of the gap between F-1 and the adapter on F-1 (26.1% against 26.4%), so a better raw model may or may not help once the list is in play. Here I run lab test 24's scoring on the pretrained adapter and compare.

## Setup

Per speaker: the fresh pretrained adapter from lab test 28 (reused if still on disk, retrained otherwise), then `phrase_score.py` with lab test 24's 15-phrase shortlist and phrase pool, then `phrase_report.py` over both systems. Thresholds come from the other two speakers.

## Plan

1. Fresh pretrained adapter, decoded in NeMo.
2. Acoustic scoring of the shortlist for every test clip.
3. The report, with lab test 24's adapter-on-F-1 scores as the comparison.

Decision rule, set before the run: the bootstrap interval of the paired difference against the adapter on F-1 at 80% coverage. Clearly lower means the raw gain carries through and I'd ship the pretrained adapter with the list. No difference means the list is doing the work and the raw model matters less on the device.

## Why

The expo device will use a list, so this is the number that matters for the model it ships.

## Limits

This gives the device reading for the best raw system, with lab test 24's honest threshold choice. It can't show a real user's list coverage, since TORGO's prompts are shared across speakers.

## Results

Behind a phrase list the pretrained adapter is no better than the adapter on F-1, even though it's 5 points better raw. The full list has 567 phrases built from all speakers' prompts. Pooled WER over 425 words (lower is better):

| System | Raw | Full list, text, no rule | Full list, acoustic | 90% coverage | 80% coverage | 50% coverage |
| --- | --- | --- | --- | --- | --- | --- |
| pretrained adapter (seed 17, fresh) | 41.4% | 30.8% | 31.5% | 33.6% | 35.7% | 41.9% |
| adapter on F-1 (lab 24) | 46.4% | 24.7% | 26.4% | 29.1% | 31.8% | 40.0% |

The acoustic and coverage columns use settings chosen on the other speakers. At 80% coverage the pretrained adapter is at 35.7% [30.5, 42.3] and the adapter on F-1 at 31.8% [26.1, 38.6]. The intervals overlap, so the list doesn't separate them, but the 5.0-point raw lead has become a nominal 3.9-point deficit. The list's gain over raw at 80% coverage (the decision interval) is −5.7 [−10.2, −1.4] for the pretrained adapter and −14.5 [−20.3, −8.5] for the adapter on F-1.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

I didn't write a prediction before starting the `1020-new` queue, and I haven't added one afterwards. The command, `python lab.py queue 25 29 28 30 27 26 --stamp 1020-new --detach`, is recorded in `app/scripts/asr/jack_notes.md` on 20 October.

Numbers from earlier runs, for comparison, taken on 21 October before any result of this test existed. Lab test 24's adapter on F-1 got 26.4% with a full list, 31.8% at 80% coverage and 40.0% at 50%. Raw, the pretrained adapter gets 178.0 of 425 words wrong against 197, so 19 fewer. Most of that gap is on sentences, which the list already fixes well (2.6% WER for the adapter on F-1 on listed sentences in stage W), so the list may shrink it. Single words were 72% to 69% WER with acoustic scoring in lab 24, so any difference would have to come from there.

### Commands and Setup

```bash
python lab.py run 30
```

About 3 adapter trainings and 3 scoring passes, so roughly 30 minutes. Results go to `reports/lab-30-phrase-scoring-pretrained/report/results.json`.

### Full Results

Source: `logs/lab-1020-new-30.txt`. The stage printed the report, and `report/results.json` stayed on the lab computer. Per speaker, the pretrained adapter scored, raw and then with the full list (text, no rule): M01 40.2% and 29.0%, M02 41.1% and 27.4%, M04 42.9% and 34.8%.

With settings chosen on the other speakers at 80% coverage and applied to each held-out speaker, acoustic scoring gave M01 35.0% (raw 40.2%), M02 33.3% (41.1%) and M04 37.8% (42.9%). Text scoring gave M01 33.7%, M02 31.6% and M04 38.0%.

By kind of clip at 80% coverage (acoustic, then text), over 303 sentence words and 122 single-word words:

| System | Sentences raw | Sentences with list | Single words raw | Single words with list |
| --- | --- | --- | --- | --- |
| pretrained adapter | 27.7% | 19.2% / 17.6% | 75.4% | 76.6% / 77.7% |
| adapter on F-1 (lab 24) | 36.0% | 17.0% / 17.2% | 72.1% | 68.7% / 72.5% |

Both systems end near 17 to 19% on sentences, so the pretrained adapter's 8-point raw lead there is gone. The difference behind the list is in the single words, where the pretrained adapter is about 6 (text) to 10 (acoustic) words worse out of 122, and the list gains it nothing.

The other speakers' data covers 100% of the test prompts for M01, M02 and M04, so the full-list rows are the optimistic case and the 80% and 50% rows are mixtures built from that list. Thresholds weren't tuned on the test clips except in the "for reference only" line (best 34.9%, text, t≤0.70).

### Conclusions

For a device that scores against a phrase list, the raw ranking isn't the deciding number. In this run the adapter on F-1 is as good or better behind the list. With one seed each and a 6 to 10 word gap on single words, that doesn't prove the adapter on F-1 is better, but it does remove the pretrained adapter's apparent lead. On sentences, which are all the prototype displays, the two end at the same level, about 17% at 80% coverage.

</details>

## Related

[Lab test 24](../lab-24-phrase-scoring/notes.md), [Lab test 28](../lab-28-deployable-export/notes.md)
