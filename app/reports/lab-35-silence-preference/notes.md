# Lab test 35: does the model prefer silence, and does the best fine-tune remove that?

Proposal: F-2 (extension), a diagnostic following [lab test 6](../lab-06-reference-score/notes.md). Prototype bar: 35%, which is 148 wrong of 425 pooled test words.

Status: written 5 November 2026. Not run, no prediction yet.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

Lab test 6 found that stock Parakeet and the whole-block models score silence as more likely than the true words on 81% to 100% of the severe speakers' test clips. That's my best explanation for why severe speech comes out empty or short, and nothing in the literature sweep covers it, so a clean measurement would be new.

This test fixes three gaps in lab test 6:

1. Formatting. The true words were scored lower case with no punctuation, but stock writes "Tomato.", so part of the 94% to 100% could be formatting. Here each system is scored with the true words in stock's own form (capital, closing mark) and in scoring form. The cost of silence doesn't depend on form.
2. No comparison. Stock is also scored on the TORGO control speakers, to see if this is specific to dysarthric speech.
3. Stage P's pretrained adapter gives almost no empty answers (0 of 166 in NeMo, lab test 28) against whole-block's 24. Scoring it shows whether it changed the preference or just greedy decoding.

## Setup

Stock is Parakeet-TDT-0.6B-v2, never trained on TORGO: 323 of 425 wrong (76.0%), 34 empty answers of 166 clips. Whole-block is the F-2 model training encoder blocks 0 to 7 (`models/lab-baselines/whole-block-<speaker>`, retrained if cleaned, so it can differ from lab test 6 by a few clips): 291 wrong, 24 empty. Stage P's adapter is a model pretrained on the other severe speakers plus an adapter after block 8 at 6e-4 on the target speaker (seed 17, retrained fresh as in lab test 28): about 176 to 183 wrong, 0 empty in NeMo. Controls are the TORGO speakers without dysarthria, test split, stock only.

The margin is the cost of silence minus the cost of the true words, in nats (log probability). Below zero means the model prefers silence. The share preferring silence is the fraction of clips below zero. Lab test 6's numbers (scoring form only) for comparison:

| Speaker, model | Clips | Median margin | Clips preferring silence |
| --- | ---: | ---: | ---: |
| M01 stock | 65 | -6.6 | 94% |
| M01 whole-block | 65 | -5.5 | 92% |
| M02 stock | 37 | -8.8 | 95% |
| M02 whole-block | 37 | -4.5 | 81% |
| M04 stock | 64 | -7.9 | 100% |
| M04 whole-block | 64 | -8.1 | 97% |

## Plan

1. Rebuild stage P's pretrained models and block-8 adapter if cleaned.
2. Per severe speaker, score stock, whole-block and stage P's adapter on the test clips, true words in both forms plus each model's own answers.
3. Score stock on the controls' test clips, both forms.
4. Print one table: speaker, system, form, median margin, share preferring silence, empty answers.

How I'd read it (written before the run). If controls rarely prefer silence and severe speakers mostly do in both forms, it's about how the model hears severe speech, and that's the finding. If the stock-form scores go near zero or flip, lab test 6 was mostly formatting. If stage P's adapter is above zero on most clips, the best fine-tune works mainly by removing the silence preference. If it still prefers silence but gives no empty answers, the fine-tune changed greedy frame-by-frame choices without changing which whole answer scores highest.

No decision rule, it's a diagnostic. I just record the shares per speaker, model and form.

## Why

It's the one result here that explains a mechanism: why severe speech loses words, and what the winning recipe changed. It also checks whether lab test 6's main number was a formatting artefact.

## Limits

Three models on 425 severe test words, plus stock on typical speech. Nothing about the intended user, other dysarthric speakers or other recognisers. Three severe speakers, one corpus.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

Not yet written.

### Commands and Setup

Run from `app/` in the WSL tree on the lab computer after `git pull`, GPU free, PC set not to sleep. About 45 minutes if the whole-block baselines and stage P's pretrained models still have weights, up to about 2 hours if they need retraining (an estimate; scoring is a few minutes per system).

```bash
python lab.py run 35
```

The table prints at the end of `logs/lab-<stamp>-35.txt` and is saved as `reports/lab-35-silence-preference/table.json`, with one `.json` per scored system in the same folder.

### Full Results

Not run.

### Conclusions

Not run.

</details>

## Related

[Lab test 6](../lab-06-reference-score/notes.md), [Lab test 28](../lab-28-deployable-export/notes.md), [Stage P](../f2-adapter-levers/notes.md)
