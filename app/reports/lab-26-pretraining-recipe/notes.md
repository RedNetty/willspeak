# Lab test 26: the pretraining recipe under the adapter

Proposal: F-2 (extension), following stage C, stage P and lab test 11. Prototype bar: 35%, which is 148 wrong of 425 pooled test words.

Status: written 20 October 2026, run in the `1020-new` queue (finished 02:09 on 21 October, lab clock). Results read from the printed tables in `logs/` on 21 October 2026. The queue's own tables and reports stayed on the lab computer.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

Severe-speaker pretraining is the only thing that got the adapter well under 200 wrong (stage P: 178 at 6e-4 over three seeds, vs 213 from stock). But the pretrained model was never varied. It's F-1's recipe (dev-selected, up to 10 epochs, rate 2e-5, all parameters trainable) on the other two severe speakers' prompt-disjoint clips, seed 17. Stage S showed the data matters (the all-dysarthric F-1 start gave 203.6). Here I vary how it's trained.

## Setup

Five starts, each followed by the same adapter (block 8, 15 epochs, 6e-4, seeds 17, 18, 19), scored on the 425 pooled test words:

1. control: stage C's phase 1 unchanged (its adapters are stage P's `pre-ad8-lr6e-4` runs, reused)
2. bottom8: only encoder blocks 0 to 7 trainable in phase 1 (`--train-parts all`), like the historical F-1
3. lr4e-5: double the phase-1 rate
4. longer: up to 20 epochs, patience 6
5. dysarthric-then-severe: the F-1 model (all other dysarthric speakers) first, then the severe-only phase on top

Phase 1 trains on `loso-severe-<speaker>`, which leaves out the speaker's own prompts. Phase-1 weights are deleted after their adapters are decoded.

## Plan

1. Rebuild stage C's phase-1 models and stage P's control adapters where their answers are missing.
2. For each speaker and variant, train phase 1 (seed 17), then the three adapters, and decode and score each.
3. Print one pooled table with every seed and the three-seed means.

Decision rule, set before the run: the usual rule on the three-seed mean vs the control (2 of 3 speakers the same way, more than about 4.5 points, 19 words). A win would make the recipe worth sweeping. All within the rule would mean the start is saturated and the data matters more than the training. If bottom8 matches the control, the full fine-tune isn't needed and the Pi export stays smaller.

## Why

It's the only lever with a big effect, and its settings were fixed once, by accident of stage C's design.

## Limits

Three seeds per arm. Nothing about a fourth severe speaker or the intended user.

## Results

Nothing beat the control. Same adapter on every arm (block 8, 15 epochs, 6e-4), only the start differs. Wrong of 425:

| Pretrained model | 17 | 18 | 19 | Mean | WER |
| --- | --- | --- | --- | --- | --- |
| control: severe speakers, every parameter (stage P) | 183 | 176 | 175 | 178.0 | 41.9% |
| bottom 8 blocks only | 200 | 184 | 217 | 200.3 | 47.1% |
| rate 4e-5 | 174 | 186 | 185 | 181.7 | 42.7% |
| longer (20 epochs, patience 6) | 182 | 179 | 180 | 180.3 | 42.4% |
| dysarthric speakers first, then severe | 185 | 188 | 183 | 185.3 | 43.6% |

Bottom 8 only is worse: 22.3 words above the control, past the 197 loss line, and worse on every speaker (summed over seeds, M01 254 vs 223, M02 138 vs 111, M04 209 vs 200). The other three arms are 2 to 7 words off, which is seed noise. Seed range within an arm runs from 3 words (longer) to 33 (bottom 8). Nothing reached the 159 gain line, so stage P's recipe stays.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

I didn't write a prediction before starting the `1020-new` queue (the command `python lab.py queue 25 29 28 30 27 26 --stamp 1020-new --detach` is in `app/scripts/asr/jack_notes.md`, 20 October), and I haven't added one afterwards.

Numbers from earlier runs for comparison, collected 21 October before any result came in:

- The control is stage P's `pre-ad8-lr6e-4`: 183, 176, 175 at seeds 17 to 19, mean 178.0. The rule needs about 19 words, so a mean of 159 or less is a gain and 197 or more a loss.
- Earlier pretrained starts under the same kind of adapter: stage C's `pre-ad8` (default adapter rate) 184, 182, 181, mean 182.3, and stage P's default-rate run 185.0. Lab test 11's whole-block start with no adapter was 248.
- The F-1 model alone is 248 wrong. The adapter on F-1 is 197 (seed 17) and 203.6 over five seeds (stage S). The dysarthric-then-severe arm starts from that model.
- Median seed spread over 51 recipe families is 10 words, so differences between three-seed means under about 8 words are noise.

### Commands and Setup

```bash
python lab.py run 26
```

About 12 phase-1 trainings (about 8 minutes each) and 36 adapters (about 1.5 minutes each), so roughly 3 hours on the RTX 4000 Ada. Results go to `reports/lab-26-pretraining-recipe/table.json`.

### Full Results

Source: `logs/pool_table-20261021-070934-018529.log`. Wrong words per speaker (M01 of 169, M02 of 95, M04 of 161):

| Arm and seed | M01 | M02 | M04 | Pooled | WER |
| --- | --- | --- | --- | --- | --- |
| stock | 131 | 62 | 130 | 323 | 76.0% |
| adapter on F-1 | 78 | 47 | 72 | 197 | 46.4% |
| control, 17 | 74 | 37 | 72 | 183 | 43.1% |
| control, 18 | 77 | 37 | 62 | 176 | 41.4% |
| control, 19 | 72 | 37 | 66 | 175 | 41.2% |
| bottom 8, 17 | 86 | 43 | 71 | 200 | 47.1% |
| bottom 8, 18 | 74 | 42 | 68 | 184 | 43.3% |
| bottom 8, 19 | 94 | 53 | 70 | 217 | 51.1% |
| rate 4e-5, 17 | 70 | 37 | 67 | 174 | 40.9% |
| rate 4e-5, 18 | 78 | 44 | 64 | 186 | 43.8% |
| rate 4e-5, 19 | 76 | 46 | 63 | 185 | 43.5% |
| longer, 17 | 79 | 35 | 68 | 182 | 42.8% |
| longer, 18 | 73 | 37 | 69 | 179 | 42.1% |
| longer, 19 | 76 | 37 | 67 | 180 | 42.4% |
| dysarthric then severe, 17 | 76 | 39 | 70 | 185 | 43.5% |
| dysarthric then severe, 18 | 78 | 43 | 67 | 188 | 44.2% |
| dysarthric then severe, 19 | 76 | 38 | 69 | 183 | 43.1% |

Every arm beats stock with an interval clear of zero, and none meets the bar (148). The best single run (rate 4e-5, seed 17, 174) is one seed of an arm whose other two scored 186 and 185.

### Conclusions

Stage P's pretraining (all parameters trained on the other severe speakers, default settings) is the best of the five starts. Changing its rate, length or order didn't help, and limiting it to the bottom 8 blocks hurt. I'm closing this line for the prototype, since more pretraining sweeps would just be chasing seed noise.

</details>

## Related

[Stage P and the adapter levers](../f2-adapter-levers/notes.md), [Lab test 11](../lab-11-severe-pretraining/notes.md)
