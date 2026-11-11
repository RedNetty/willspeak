# Lab test 39: do candidate answers from other models raise the ceiling, and does the same chooser find them?

Proposal F-2 (extension) and H4 (every extra decode costs time on a Raspberry Pi 5). I measure whole sentences right, with sentence WER next to it, and decodes per clip next to every gain.

Status: written 9 November 2026. Prediction not written yet.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md), [Lab test 36](../lab-36-lm-rescoring/notes.md)

## Idea

[Lab test 36](../lab-36-lm-rescoring/notes.md) decodes each clip nine ways with stage P's adapter, pools the distinct answers, and lets a small text model plus the recogniser's audio score pick one. The chooser can only pick something a decode proposed. On stage P's recipe over the 120 labelled sentences, lab 36 reported the right sentence in the pool for 45 (37.5%), the first decode right for 29 (24.2%), and the chooser at 31.7 to 35.0% for most language models, with 3.9 answers per clip. (Its table isn't on this branch.) So the chooser is about 10 sentences short of the ceiling.

The nine decodes come from one model, so they tend to agree. A different model might propose the answers they miss. This test adds answers from other models on the same pretrained base that already exist or are quick to build:

1. seeds: stage P's adapter retrained with seeds 18 and 19 (labs 25 and 34 trained these three seeds; the primary is seed 17)
2. positions: the adapter after block 12 and after block 16, seed 17 (lab 34's recipe, `stage_34`)
3. both: all four extra models in one pool

Each extra adds its greedy decode, optionally plus a 4-wide beam (`wide_extra_views`). Every answer is scored on the audio by the primary model (stage P's adapter, seed 17, block 8) and as English by the same text model and weights, so only the proposers change between arms. The language model is `qwen25-05b-q4`, lab 36's primary one (`wide_lm_models` can name others; the table is for one).

The nine views are nine decodes but seven encoder passes (greedy and the two beam searches share one). Each extra adds a decode and a pass, so the arms cost 11, 11 and 13 decodes a clip, recorded next to every gain.

## Setup

The primary model is lab 36's: Parakeet-TDT-0.6B-v2 trained on the other severe speakers (`big-c-pre-<speaker>`, which never saw the target speaker), then an adapter after block 8 at 6e-4 on the target speaker's clips (15 epochs, seed 17). The extras change only the seed or the block. Decoding is NeMo at full precision.

Each arm gets two systems, built exactly like lab 36's two stage P tables:

| System | Clips | Models |
| --- | --- | --- |
| stage P's adapter, the test clips | 166 clips of M01, M02 and M04: 44 sentences, 122 single words | primary and extras trained on train and dev |
| stage P's recipe, every labelled clip held out | 513 clips: 120 sentences, 393 single words | the test clips as above, plus four prompt-grouped folds per speaker; for each fold the primary (`lab-36-pre-ad8-<speaker>-f<k>`, lab 36's name) and every extra are trained on the other folds' clips and decode the held-out fold |

An extra model never decodes a clip it has seen. For the test clips the extras train on `export/<speaker>` (train and dev only; `train.py` never opens the test split). For a fold they train on `export/cv-<speaker>-f<k>`, like that fold's primary, and carry the fold in their name (`lab-39-pre-ad8-<speaker>-f<k>-s18`, `lab-39-pre-ad12-<speaker>-f<k>`). Three checks enforce this:

1. `wide_leaks` in `lab.py` refuses a unit whose models are named for another fold, or for the test clips inside a fold's pool (the likeliest mistake). `test_wider_pools.py` fails if that guard is weakened. It also checks each planned training command's `--export` against its name, and each planned pool command against its extras.
2. `wide_unseen` reads the manifests at run time and stops the unit if a held-out clip id, or for a fold a held-out prompt, is in the training clips.
3. The adapter recipe is `--select last --no-keep-stock`, and the stage won't start if `xflags` changes either flag. In a cross-validation export `dev.jsonl` is the held-out clips, so picking weights on dev would leak. Lab 36's fold adapters rely on the same thing.

### Pools and the 15-nat rule

Each arm's pool is lab 36's baseline pool plus the extras' answers. Baseline candidates are copied unchanged (text, audio score, order, first decode first). An extra answer that matches one (ignoring capitals and punctuation) only adds its tag to that candidate's `views`. A new answer is scored by the primary model and appended. If two extras give it in different forms, each is scored and the best kept, as in lab 36, and the forms are saved so narrower arms can be cut exactly.

The 15-nat drop rule is measured against the best audio score among the primary model's own answers. Against the merged best, an extra answer the primary liked a lot could push baseline answers out, maybe the right one. As built, every arm contains the whole baseline pool, so the ceiling can only rise or stay put. Tags look like `seed18` or `block12/beam4`, and "extra-only" below means every view that proposed the chosen answer is an extra model.

The widest pool is built once, one model on the GPU at a time, and the narrower arms are cut from it without decoding (`lm_pool.py --subset-of ... --keep-extras ...`). A test checks the cut matches what only those models would produce.

`reports/lab-36-lm-rescoring/pools/prototype-*.json` is reused as the baseline only if the primary adapter on disk is the one that scored it (its recorded SHA-256 matches). Lab 36 deletes fine-tuned models when disk is short on the lab computer, and a retrained adapter gives different audio scores under the same name. Then this test rebuilds the baseline under `reports/lab-39-wider-pools/pools/` (nine decodes per clip, a few minutes per speaker), logs it, and leaves lab 36's folder alone. That baseline is lab 36's recipe retrained, so it can differ from lab 36 by training noise. Language-model scores live in this test's folder, seeded from lab 36's score file when it exists so nothing is scored twice.

### What the report prints

Per arm and system, the same table as lab 36 (first decode, audio score only, language model, pool ceiling, paired intervals, gained and lost, sentence WER, single words right, preset). The heading adds answers and decodes per clip, gained sentences whose answer came only from an extra model, and clips whose right answer is in the pool only thanks to an extra. Then a block of paired changes against the baseline arm for the first decode (a check, must be zero), audio score, language model and ceiling, each with a 95% interval resampling clips and one resampling prompts.

For comparison: seeds 17, 18 and 19 of stage P have nearly the same word errors on the 425-word test set (lab 34 block 8: 176, 172, 175 wrong). How often they disagree on a clip isn't recorded, which is what the seeds arm measures.

## Plan

1. `python lab.py queue widepool --dry` prints the plan.
2. Per speaker, for the test clips and each of four held-out folds: train the primary (or reuse it) and the needed extras, build or reuse the baseline pool, decode the extras once into the widest pool, cut the narrower arms, and remove the adapters if disk is short.
3. Score every new answer with the language model once.
4. Write one report with the tables and paired comparisons.

Decision rules, fixed before any run:

An arm is judged on "stage P's recipe, every labelled clip held out" (120 sentence clips) with `qwen25-05b-q4`, from the paired block against the baseline arm. It counts only if (1) the pool ceiling rises (the "best answer in the pool" row) and (2) the chooser's sentences right rise (the language-model row), each with the 95% interval resampling clips above zero. If the by-prompt interval includes zero, it's reported as "counts by clips, not by prompts", since several speakers read the same sentence. The arm also must not lose sentences on the test-clip system, and single words right must not lose more clips than the sentences gain.

Cost goes next to the result: decodes and encoder passes per clip, and sentences gained per extra decode. An arm is only worth it if it beats a bigger pool from the same model, which this test can't check, so the result note has to say that. I also report, without ruling on it, how many gained sentences came only from an extra model. If almost none did, the extras just changed which wrong answers the chooser sees, and rule (1) shows whether any right answer was added. The lowest-cost arm that counts goes forward. If only "both" counts, the extras only add up in combination. Nothing changes after the first table (grids, preset, language model, extras, views), and other language models (`wide_lm_models`) are only a screen, as in lab 36.

What the outcomes would mean, written before the run. If the ceiling and the chooser both rise, same-base models do propose answers the nine views miss and the two scores can find some. Next would be seeds against positions per gained sentence, and timing on the Pi. If the ceiling rises but the chooser stays flat or drops, the chooser can't separate the new right answers from the new wrong ones, so the chooser is the problem (a stronger language model, a trained selector). If the chooser actually loses sentences, the weights chosen on other speakers don't transfer across pool sizes. A flat ceiling means the extras mostly repeat the pool, and diversity has to come from other training data or bases, or the views themselves. Seeds working and positions not (or the reverse) points at where the diversity is: the adapter's random initialisation or the depth where the speaker's voice gets absorbed. A gain only in "both" means no single cheap extra is worth its decode, and the cost may be more than the device can afford. If no arm counts, the nine-view pool is already as wide as same-base models make it, and lab 36's result stands alone.

## Why

If the right sentence is in the pool for 37.5% of sentences and the chooser gets 32 to 35%, there isn't much gap left, so the bigger prize is raising the ceiling itself. The easy candidates are models that already exist or take minutes to build. Whether any of this is usable depends on the device, since each extra model is another decode on a Pi 5. What matters is gain per decode.

## Limits

This shows whether other seeds and adapter depths of stage P's recipe add right sentences to the pool, whether the unchanged chooser finds them, and the cost in decodes and encoder passes per clip.

It doesn't show time or memory on the Pi. There every extra model is another adapter run, though blocks 1 to 8 are identical in all of them and could be computed once (not measured here). It says nothing about a model trained on the intended user's own speech (the TORGO speakers are adults), or whether a longer pool from one model would do as well. The sentences are TORGO prompts shared across speakers, hence the by-prompt interval, and the test clips have been scored in many earlier experiments. The extras share a base and recipe with the primary, so they aren't independent draws and a gain wouldn't carry over to unrelated models. Since the primary model scores every answer, an extra model's own confidence is never used.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

Not written yet.

### Commands and Setup

All times below are estimates. Nothing has been run.

Adapters train on the GPU through the usual lab path (`ensure_pretrained_models`, `train_adapter_arm`). Each pretrained base frees its checkpoints before the next one trains, as in lab 36's disk fix. The language-model environment (`~/venvs/willspeak-lm`) is built when missing, as in lab 36.

Settings: `wide_extras` picks which arms to build besides the baseline (`seeds`, `positions`, `both`; default all three). `wide_systems` is `prototype` and/or `folds` (default both). `wide_extra_views` defaults to `greedy`, and `greedy beam4` adds a beam search per extra model. `wide_lm_models` defaults to `qwen25-05b-q4`. Lab 36's `lm_views`, `lm_threads`, `lmpy`, `cv_folds` and `trim_models` also apply.

A first look at the test clips only (about 15 adapters, about 1 hour):

```bash
python lab.py --set wide_systems=prototype run 39
```

Everything (default: all three arms, both systems):

```bash
python lab.py run 39 --rerun          # or: python lab.py queue widepool --detach
```

`--rerun` moves an earlier `table.json` aside, and isn't needed if there isn't one. `python lab.py --set wide_extras=seeds run 39` builds only the cheapest arm (two extra models). On the lab computer, `python lab.py --set push_results=on queue widepool --detach` keeps running after the terminal closes and saves the results on a `lab-results-*` branch. There's no `willspeak.sh` wrapper for this test. `python lab.py queue widepool --dry` prints the plan without the GPU.

With the defaults that's 75 adapter trainings on the 4060: 15 primaries (lab 36 may have left them on disk) and 60 extras. Lab 33's 45 adapters took about 2 hours with decodes, about 2.7 minutes each, and these are lighter, so about 3 hours. Add 15 baseline pools if lab 36's can't be reused and 15 wider pools (a few minutes each, mostly loading five models), and about 20 minutes of language-model scoring. About 4 to 5 hours in all, plus about 25 minutes (lab 34's estimate) if the three `big-c-pre-<speaker>` bases need retraining. `wide_extras=seeds` roughly halves the training.

Each adapter run writes top-4 checkpoints (`ckpt/`) besides `adapter.pt`. The stage deletes `ckpt/` right after training, since five adapters are alive at once per clip set and nothing reads them again.

The tables are printed at the end of `logs/lab-<stamp>-39.txt` and saved as `reports/lab-39-wider-pools/table.json`. Pools are in `pools/` (`base-*` when lab 36's couldn't be reused, `wide-*` for the widest pool, `seeds-*` and `positions-*` for the cut arms), and the language model's scores are in `scores/`.

### Full Results

Not run.

### Conclusions

Not run.

</details>

## Related

[Lab test 36](../lab-36-lm-rescoring/notes.md), [Lab test 34](../lab-34-pretrained-adapter-positions/notes.md), [Lab test 25](../lab-25-tta-on-pretrained/notes.md), [Stage P](../f2-adapter-levers/notes.md)
