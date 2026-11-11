# The semester plan

Paths and commands below are relative to `app/` unless they refer to documentation.

- [Fall 2026 plan, the public-corpus semester](#fall-2026-plan-the-public-corpus-semester)
- [Storyboard outline (due Sep 27; presented Sep 28)](#storyboard-outline-due-sep-27-presented-sep-28)
- [Ethics review and the data this semester uses](#ethics-review-and-the-data-this-semester-uses)
- [Recording protocol](#recording-protocol)


---

## Fall 2026 plan: the public-corpus semester

*Written 6 September 2026; revised 10 September to add the Parakeet
fine-tune on TORGO at my decision. Supersedes the August roadmap's
milestone list for this semester; the design
([design.md](design.md)) stands.*

### What changed and why

The professor placed this project under the university's human-subjects
review, which means recording the client this semester would need IRB
training and an approved protocol first. Neither fits in the ten weeks
between now and the prototype, so this semester's project **uses public
dysarthric speakers instead of the client**: TORGO, a published,
de-identified research corpus of eight speakers with cerebral palsy or ALS.
Analysis of public de-identified data is not normally human-subjects
research at all ([Ethics review and the data this semester uses](#ethics-review-and-the-data-this-semester-uses) has the reasoning and the
one email to send). The client stays in the project, he is its reason, but
everything that involves his voice moves to a later phase that starts only
after the IRB work is done.

The course therefore sees a complete system whose accuracy claims are measured
on public speakers I have never met, including **a Parakeet model fine-tuned
on one severe speaker's own recordings, tested on prompts it never saw,
running on the Raspberry Pi 5**. In the demo I speak (deliberately unclearly)
and the box speaks back.

**Every number on the storyboard or the poster must come from a recorded
run I made on the stated hardware, saved under `reports/`.**
The Round 3 CPU pilot numbers are context, not results.

**10 September revision.** The 6 September plan left fine-tuning out of the
semester. I reversed that: without a fine-tuned model there is no number
that shows the architecture's advantage, and the personalisation finding is
the project's whole premise. The TORGO fine-tune is designed in
[research/finetune-torgo.md](research/finetune-torgo.md); I run the experiments
and interpret and present their results.

### Course deliverables (ENGR 1113)

| date | deliverable | weight | what it needs from this plan |
|---|---|---|---|
| Sun Sep 27, midnight | Storyboard digital submission, Gradescope | 20% of the storyboard (15 pts) | team name, members, the findings written out (context, problem, solution) and a picture of the storyboard, as one PDF |
| Mon Sep 28, 1 pm | Storyboard physical, presented to the freshman class; token investment | 80% (100 pts); the whole storyboard is the 30% design review | 3–8 scenes, each at least 8.5 × 11 in, on a poster from the Fields 205 closet; rubric: story (who-why-what), investments, presentation and creativity, ≥ 2 clearly defined research findings ([Storyboard outline (due Sep 27; presented Sep 28)](#storyboard-outline-due-sep-27-presented-sep-28)) |
| ~Oct 19, ~Oct 26, ~Nov 3 | Prototype Preparation reports, one before each stage (basic, intermediate, advanced), Gradescope, 2 pages max | - | Overview; Methodology (circuit with part numbers, code design as a state-flow chart); Experimentation and Testing (setup, protocols, repeatability); Results and Analysis. Dates shifted from last year's 10/21, 10/28, 11/06,confirm on Gradescope |
| ~Oct 20 | Extra parts in hand | - | anything bought for the prototype must arrive two weeks before the expo. No display to buy: the expo view is VNC from a laptop on battery (decided 23 Sep); a USB speaker only if the voice currently comes out of the HDMI monitor |
| Tue Nov 3 | Prototype Expo (the 60% design review) | - | a working prototype at a station for the whole session, computer or display on battery (no power cord; WillSpeak: laptop on battery, VNC into the Pi); rubric: ≥ 2 client needs clearly addressed, 2 major improvements from first to final prototype, clarity, functionality (the client can use it easily), investments. Afterwards each student writes up 3-4 other stations on Canvas |
| ~Mon Nov 16, 3 pm | Poster file for printing, Canvas | one submission only | 36 × 36 in, white background, from the course PowerPoint template, saved as PDF; print it on letter paper first to proofread. Deadline shifted from last year's Mon Nov 17,confirm on Canvas |
| Thu Nov 19, 1:20 pm | Final poster presentation, judged by outside engineers | - | Title and names; Objective (problem, goal, impact); Research that produced real constraints; Design (methodology, electrical and code design, sensors and actuators, pictures, testing and analysis that changed the design); Future of the project with sustainability as a strong objective. Every measurement in [measurements.md](measurements.md); project scope and limitations |

Dates are from the 2026F storyboard assignment (storyboard, prototype, poster);
the prototype-report and print dates are last year's documents moved to this
year's calendar, so check Canvas and Gradescope for those. The course's
standard track is an Arduino alert system for a low-income community;
WillSpeak is my approved project instead (confirmed 15 Sep), so the
"circuit" and "sensors and actuators" sections map to the Pi 5, the lavalier
microphone, the speaker and the power bank, with the power measurements as the
electrical design.

### Scope

**In:**

- TORGO as the corpus for every accuracy claim: zero-shot Parakeet WER on the
  severe speakers, **Parakeet fine-tuned per speaker on the same speakers with
  prompt-disjoint splits (F-1, F-2)**, and the matcher's closed-vocabulary and
  cross-session accuracy on their words, all measured with the
  scripts in this repository ([training.md](training.md#public-corpora-torgo)).
- The Raspberry Pi 5 as the prototype box ([pi5.md](pi5.md#the-deploy-folder)):
  bring-up, speed, latency, cold start and stop-phrase reliability, all measured on the board, and the fine-tuned model's speed on it.
- The live demo with my voice (stock model; a model fine-tuned on
  F01 does not establish a training benefit for my voice). Recording myself to demonstrate
  my own device is not research on other people; it still gets a sentence
  in the IRB email so nobody is surprised.

**Out (moved to the IRB-gated phase, [Recording protocol](#recording-protocol)):**

- Any recording of the client or his family; the consent gate; the
  personalised fine-tune on *his* speech; the "how is he today" policy
  fine-tuned on his sessions.
- UASpeech, EasyCall, SAP: more corpora would not change the story and each
  costs a licence request or an agreement. TORGO is enough for the claims
  this semester makes.
- The EasyCall results of Round 3,cited as CPU pilot context if at all.

### Week by week (about five hours a week, plus GPU waiting time)

| week | dates | work | done when |
|---|---|---|---|
| 1 | Sep 7–13 | **Bring-up.** Commit the 10 Sep cleanup after `python -m pytest` is green; the fixed pages onto the Pi; Pi bench (**P-A**, done 8 Sep). Send the IRB email ([Ethics review and the data this semester uses](#ethics-review-and-the-data-this-semester-uses)). The items are at the top of [TODO.md](TODO.md). | committed; the email is sent |
| 2 | Sep 14–20 | **TORGO and the baseline.** `scripts/corpus/fetch_torgo.py` (3.6 GB; `--controls` adds the typical-speech check), `scripts/corpus/torgo_manifest.py --export`. Zero-shot Parakeet on the severe sentences (**T-1**). WSL2 + CUDA + NeMo on the 4060 and `train.py --check` green (fallback: a Kaggle/Colab notebook). | `reports/torgo-severe-zero-shot/notes.md`; the training environment loads the model |
| 3 | Sep 21–27 | **Storyboard, and the first fine-tuned model.** The throwaway one-epoch fine-tune through the whole F-4 chain (train → export → check → Pi), then F-2 on one severe speaker (a first number for the poster panel if it finishes; T-1 alone if not). Scenes from [Storyboard outline (due Sep 27; presented Sep 28)](#storyboard-outline-due-sep-27-presented-sep-28); the two research findings; the physical poster; rehearse the two-minute pitch. | PDF on Gradescope Sep 27; poster on the table Sep 28 |
| 4–5 | Sep 28–Oct 11 | **F-1 (one held out) and F-2 (later session held out) on the severe speakers, the typical-speech check alongside** (an hour of GPU per run; evenings). Pi: end-to-end latency (**P-B**). | `reports/finetune-torgo/<speaker>/notes.md` ×4 |
| 6 | Oct 12–18 | **The matcher on public speakers.** Embed the TORGO words (`embed_manifest.py --manifest words.csv`), `train_projection.py --eval` (**T-2**), `matcher/evaluate.py --emb` (**T-3**). Pi: cold start (**P-D**). | `reports/torgo-matcher/notes.md` with the tables |
| 7 | Oct 19–25 | **Prototype Preparation report, stage 1** (before the stage starts). **The fine-tuned model on the Pi (F-4)** and the data-quantity curve (**F-3**). | `reports/bench-pi5-fine-tuned/`; the F-3 table |
| 8 | Oct 26–Nov 1 | **Prototype Preparation report, stage 2.** **Prototype hardening.** `setup.sh --service`, stop-phrase trials (**P-E**), the failure list (what happens when the mic is unplugged, the model fails to load, a wrong sentence is spoken). Demo script with one deliberate failure. | power-cycle → speaking, no keyboard; the failure table |
| 9 | Nov 2–3 | **Prototype expo, Tue Nov 3.** Rehearse in a noisy room; freeze `WILLSPEAK_ASR_THREADS` and the thresholds; Pi on the power bank, laptop on battery, VNC over a phone hotspot. Stage 3 report. | the demo ran; what broke is written down |
| 10–11 | Nov 4–19 | **Poster.** Every table in [measurements.md](measurements.md) from `reports/`; a clear account of completed work and its limitations; Future panel with sustainability. File to Canvas ~Mon Nov 16 (one shot), presented Thu Nov 19. | PDF submitted; poster judged |
| Dec | after the course | If the client phase is going ahead in spring: CITI training over the break, protocol drafted with the professor. | protocol draft |

If a code week runs over, take the week from the poster's slack (week 10 has
more hours in it than the poster needs), never from the prototype week. If
the training environment is not working by the end of week 3, switch to the
Kaggle/Colab fallback that day rather than fighting WSL2 into week 4.

### Project work this semester

My work covers the runs under `reports/`, Pi bring-up and measurements,
fine-tuning and predictions, the storyboard, pitch and poster, core-algorithm
exercises, the IRB email, the failure list, demo script, and CI script.
[TODO.md](TODO.md) tracks the creation work; routine cleanup stays out of it.

The tooling I operate and explain includes the TORGO fetch and
manifest tools (`src/willspeak/torgo.py`, `scripts/corpus/fetch_torgo.py`,
`scripts/corpus/torgo_manifest.py`), the `--emb` mode of
`scripts/matcher/evaluate.py`, the `--group-by`/`--limit`/machine-provenance
additions to `scripts/asr/evaluate.py`, the fine-tune recipe and the
`--freeze-encoder-layers` / early-stopping / speed-perturbation options in
`scripts/asr/train.py`, the device bench (`scripts/asr/bench.py`), the Pi
deployment folder (`deploy/pi5/`), and the documents in this folder.
The storyboard's examiner may ask
*why the microphone is at his collar and not on the chair* ([design.md](design.md#other-voices-in-the-room)), *why the splits are prompt-disjoint*
and *why the bottom of the encoder is frozen*, and the answers are in the
docstrings and in [research/finetune-torgo.md](research/finetune-torgo.md).

**Outside this semester's scope:** client audio, new corpora.

### Definition of done, per deliverable

- **Storyboard**: a stranger can read the three scenes and say what the
  device does, for whom, why existing products do not, and what two published
  findings the design rests on; at least one number comes from my recorded runs
  (T-1 at minimum).
- **Prototype**: power-cycle → the Speak page is listening within two
  minutes with nobody touching it; I speak a sentence, the box speaks it
  back; I say the stop phrase after a wrong sentence and the box retracts
  it; the whole thing survives a room with people talking; the fine-tuned
  model has been benchmarked on the same board.
- **Poster**: every number traceable to a `reports/<run>/notes.md`; the
  stock-versus-fine-tuned table with n and confidence intervals; the
  completed-work description; the limitation ("measured on public speakers, not on
  the person it is for, that phase needs ethics approval and comes next").

### Risks and what happens then

| risk | signal | fallback |
|---|---|---|
| WSL2 + CUDA + NeMo will not install on the 4060 box | `train.py --check` fine but the model will not load by end of week 3 | Kaggle (30 h/week, 16 GB) or Colab notebook; TORGO is public so this is fine for the semester; the client phase must stay local |
| The fine-tuned model is not better | F-1/F-2 Δ ≤ 0 on most speakers | that is a result: report it with the data-quantity curve, and the poster's claim becomes the matcher's gain plus the honest fine-tune finding |
| Pi 5 too slow for live use | P-A: RTF ≥ 1.0 idle, or ≥ 0.7 with the page open | the Sky-2 is the demo box (already measured usable); the Pi becomes a "next box" slide (P-A on 8 Sep said the Pi is fine: RTF 0.13) |
| TORGO download stalls | `fetch_torgo.py` fails twice | the script resumes; the F+M archives are 3.6 GB, use campus wifi, or download on the desktop and copy `data/public/torgo/` over |
| The IRB office wants a determination form anyway | reply to the email | fill it in; it needs no training; nothing in the plan waits on it |
| The professor asks for the client anyway | -| the plan already has the answer: the phase exists, it is gated, and the semester's claims do not depend on him |


---

## Storyboard outline (due Sep 27; presented Sep 28)

The storyboard is three to eight scenes, each at least letter size, following
Context (who and why) → Problem (what, including what has been tried and why
it failed, with target markers) → Solution (how, with basic requirements and
at least two research findings). It is graded on story clarity, audience
investment, creativity and visuals, and the research findings; on Sep 28 the
class invests tokens in the projects they believe in.

This file is the skeleton and the sourced facts. The drawings, the wording
on the panels and the pitch are mine. Every number marked **[T-x]**,
**[F-x]** or **[P-x]** comes from my recorded runs ([measurements.md](measurements.md));
the storyboard must not go out with a CPU pilot number in its place.

### Scene 1: Who (context)

An adult with cerebral palsy in a power wheelchair, at home, mid-conversation
with family. His speech is clear to the people who know him and hard for
everyone else, and on bad days hard for them too. He cannot press a button,
touch a screen or hold a switch. His AAC device (a tablet on the chair) can
*speak* for him if someone types, but it cannot *listen*.

The reader should come away seeing a specific person, and that the bottleneck
is one direction of the conversation.

### Scene 2: Why now (context)

The people who could help have data he cannot give them: the commercial
options need a phone app, an internet connection and an enrolment session
per phrase, and their own documentation scopes them to milder speech.
Finding to quote on the panel (verified 6 Sep 2026):

> "Voiceitt is most effective for individuals with mild to moderate speech
> impairments", and "Voiceitt will only work while online, so internet
> access is required." ($49.99/month.), voiceitt.com/faq

Google's Project Relate stopped accepting new users. No wheelchair AAC device
on the market recognises dysarthric speech at all. The gap is at the severe
end, offline, hands-free.

### Scene 3: The problem, in numbers (problem)

Two panels of evidence:

1. **Ordinary speech recognition fails on this speech.** My recorded
   measurement: NVIDIA's Parakeet recogniser, unmodified, on the sentences of
   the four severe TORGO speakers, WER **[T-1]** (n = ~290 clips). Show one
   real transcript pair from `report.html` (what was said / what the machine
   heard). Frame it with the literature: when TORGO's train/test prompt
   overlap is removed, state-of-the-art recognisers give "extremely high"
   word error rates for speakers with severe dysarthria (Hui, Zhang & Mohan,
   2024, arXiv:2411.00980).
2. **Target markers** (the rubric asks for them explicitly):
   - *accuracy*: on the words he uses every day, better than the general
     recogniser on bad days, the matcher's job; and a recogniser fine-tuned
     on *his* speech, not the average speaker's, the fine-tune's job;
   - *speed*: device speaks within ~1 s of the end of an utterance;
   - *robustness*: works across days (his voice drifts) and in a room with
     other people talking;
   - *maintainability*: a caregiver can change the voice, the delay and the
     phrases without a programmer;
   - *cost*: runs on a board under $100, no subscription, no cloud;
   - *safety*: he must be able to stop it with his voice, and it must never
     speak something it should have recognised as a command.

### Scene 4: What has been tried and why it failed (problem)

- Cloud/general recognisers: fail at the severe end (scene 3) and need
  internet on a wheelchair.
- Personalised commercial services: appointment-based data collection -
  the client tried one; a few sessions a year cannot capture bad days.
- AAC devices: output only.
- A previous attempt at this project's own personalised fine-tune on seven
  minutes of repetitive speech made accuracy *worse* (12.5% → 37.5% test WER
  on my mumble corpus, Aug 2026): repetition is not diversity, and that sets
  up the design.

### Scene 5: The solution (how)

A drawing of the box on the chair: a lavalier microphone at his collar (the
Movo USB lav; never at his face) → Raspberry Pi 5 → small speaker. Inside: two recognisers and one policy.

- **Sentence recogniser** (Parakeet-TDT, 0.6 B parameters, int8, on the CPU):
  open vocabulary, anything he says, and fine-tunable on one speaker's own
  recordings (scene 7).
- **Exemplar matcher**: his own recordings of his everyday words are the
  reference; a new utterance is compared to them (WavLM features → a learned
  projection → cosine similarity). Closed vocabulary, strongest exactly where
  the sentence recogniser is weakest.
- **The policy**: speak / stop / sleep / wake, decided per utterance; his
  only controls are spoken phrases; a stop phrase after a wrong sentence
  makes the device say "scratch that".

Basic requirements list on the panel: zero-input operation; offline; < 1 s;
caregiver-adjustable; runs on the Pi 5 at **[P-A]** × real time (measured
8 Sep 2026: 0.13×,a 3 s sentence decodes in 0.4 s).

### Scene 6: The two research findings (rubric)

Both verified at the source on 6 Sep 2026; cite them exactly.

1. **Personalisation works with little data, at the severe end.** Shor et
   al., *Personalizing ASR for Dysarthric and Accented Speech with Limited
   Data*, Interspeech 2019 (arXiv:1907.13511): 62% relative WER reduction for
   ALS speakers, to ~10% (mild) and ~20% (more serious) on message-bank
   phrases, and **71% of the improvement comes from only five minutes of
   training data.** This is why the design is built around the speaker's own
   recordings rather than a bigger general model, and why scene 7 shows
   the same effect reproduced on a public speaker.
2. **Benchmarks flatter recognisers unless prompts are kept out of the test
   set.** Hui, Zhang & Mohan, *Enhancing AAC Software for Dysarthric Speakers
   in e-Health Settings: An Evaluation Using TORGO*, 2024 (arXiv:2411.00980):
   TORGO's "well-known" prompt overlap between training and test speakers
   inflates results; with it removed, state-of-the-art models give extremely
   high WER for mild and severe speakers. This is why this project's splits
   are prompt-disjoint (`scripts/corpus/torgo_manifest.py`) and why its
   numbers are reported against that standard.

A third, optional, for the matcher panel: the corpus itself, Rudzicz,
Namasivayam & Wolff, *The TORGO database of acoustic and articulatory speech
from speakers with dysarthria*, LREC journal 46(4), 2012,eight speakers
with CP or ALS, the public stand-in for the client this semester.

### Scene 7: What is measured, and what is honest (solution)

Two panels. **The fine-tuned recogniser:** Parakeet on one severe TORGO
speaker's held-out prompts on a later day, stock **[stock]** → fine-tuned on
that speaker's own earlier recordings **[F-2]** (n, and whether typical
speech regressed), the number that says the architecture is worth
building, reproduced on a public speaker because the person it is for
cannot be recorded yet. If F-2 is not finished by Sep 27 the panel
carries T-1 and says the fine-tuned number is coming on the Nov 19 poster.

**The matcher's gain** on public speakers the projection never saw: raw
cosine **[T-3 raw]** → projected **[T-3 projected]** top-1, severe group
**[T-3 severe]** ("speakers held out; the vocabulary is shared").

The limitation panel, in plain words: *measured on eight public speakers,
not yet on the person it is for; that phase needs ethics approval and is
next.*

### Scene 8: The ask (audience investment)

What tokens buy: a Pi on a chair in November, a person who can be understood
by strangers next year. The pitch is two minutes. Suggested beats: the person
(10 s) → the sound of a general recogniser failing (play one TORGO clip and
show what Parakeet heard -20 s) → the two ideas (60 s) → the number from
T-1 (or F-2) and the promise of the fine-tuned model on the Pi (20 s) → the
ask (10 s).

### Checklist before Sep 27

- [ ] T-1 exists in `reports/` with `notes.md`, and the number on the panel
      is that one; F-2 on one speaker if the training machine is up.
- [ ] P-A is on the panel (Pi 5: 0.13× real time, my run of 8 Sep).
- [ ] Both research findings cited with authors, venue, year, arXiv id.
- [ ] Target markers panel has all six.
- [ ] The IRB email has been sent (scene 7's limitation line depends on it).
- [ ] PDF to Gradescope; physical poster printed at ≥ letter per scene.


---

## Ethics review and the data this semester uses

*This is an engineering student's reading of the rules, for the professor and
the university's IRB office to check. It is not legal advice. The one action
item is the email at the end.*

### The situation

The professor said the project falls under the university's IRB and that
recording the client would first need human-subjects training and an approved
protocol. That is the normal position: audio of an identifiable person with a
medical condition, collected to develop and evaluate a system, is human-subjects
research, and the recordings are biometric data.

So this semester's work uses data the rule does not cover, and the client
work happens later, with the approvals in place.

### Why public, de-identified corpora are different

Under the U.S. Common Rule (45 CFR 46.102), research is human-subjects
research when the investigator obtains information *about living individuals*
through interaction with them, or obtains *identifiable* private information.
Secondary analysis of data that is already public and stripped of identifiers
is generally treated by university IRBs as **not human-subjects research** -
no review, no training requirement, provided the researcher does not
interact with the people, cannot re-identify them, and does not merge sources
in a way that would. Most IRB offices publish this guidance and offer
a short "not human subjects research" determination form for anyone who
wants it in writing.

TORGO fits:

- **Published and open.** The University of Toronto distributes it from a
  public web page; "use of this database is free for academic (non-profit)
  purposes"; the only condition is citation.
- **De-identified.** Speakers are labelled F01, M04, and so on; no names, no
  faces (the video was never released), no contact.
- **No interaction.** Nobody is recruited, recorded, or contacted.
- **No re-identification.** The project never links TORGO to any other
  source.

Two caveats. A voice is in principle biometric, and
some institutions treat any speech data with extra care, hence the email
below rather than a silent assumption. And "free for academic use" is a
licence condition, not an ethics finding; the citation obligation is met by
`data/public/torgo/SOURCE.md` and by the poster.

Fine-tuning a model on TORGO does not change the analysis: it is still
secondary analysis of public de-identified data, and the resulting model is
not shared beyond the project. If the training runs on a cloud notebook
(Kaggle/Colab) that is fine for TORGO and would **not** be fine for the
client's recordings, those stay on the desktop.

### What the semester does and does not do with people

| activity | human-subjects research? | what we do |
|---|---|---|
| Evaluating, fitting and fine-tuning on TORGO | no (public, de-identified, no interaction) | ask for a written determination anyway |
| Using my voice with the device for the demo | no, using my own prototype; nothing is collected about anyone else | mention it in the email so it is on record |
| Classmates trying the demo on Nov 3 | not research if nothing is stored: the box transcribes and speaks, and the live clips are not kept | run the demo with `WILLSPEAK_DATA` pointed at a scratch folder that is wiped afterwards |
| Recording the client, his family, or any other person | **yes** | not this semester; see below |

### The client phase, later

Everything that needs the client stays in the repository as a documented,
gated phase: the [Recording protocol](#recording-protocol), the transport
loop in the [two-machine runbook](training.md#laptop-collection--desktop-training--laptop-testing), the personalised
fine-tune (the TORGO experiment in
[research/finetune-torgo.md](research/finetune-torgo.md) is its dress
rehearsal) and the session-drift policy in [TODO.md](TODO.md). None of
it runs until:

1. I have completed the university's human-subjects training (usually the
   CITI course, a few hours, online; a sensible winter-break task if the
   spring is going ahead);
2. a protocol with a faculty PI is approved (the professor said this is his
   territory);
3. consent is signed and stored before the first recording, as the protocol
   already requires.

The design does not change for the wait. The matcher's projection, the
stop-phrase machinery, the export splits, the recorder and the fine-tune
pipeline are all being validated on TORGO now; the client's data, when it
comes, plugs into the same pipeline.

### The email (send in week 1, copy the professor)

> Subject: Determination request, secondary analysis of a public
> de-identified speech corpus (ENGR 1113 project)
>
> I am a first-year engineering student building a speech-recognition
> prototype for people with dysarthria as my ENGR 1113 project, supervised
> by Prof. ___. This semester the project will **only** use the TORGO
> database, a publicly available, de-identified research corpus of
> dysarthric speech published by the University of Toronto for academic use
> (Rudzicz et al., 2012), to evaluate and fine-tune a speech-recognition model.
> I will not recruit, record, or contact any participant, and I will not
> attempt to identify anyone in the corpus. The only other audio involved is
> my own voice, used to demonstrate the device.
>
> My understanding is that this does not meet the definition of
> human-subjects research under 45 CFR 46.102. Could you confirm, or tell me
> which form to submit for a determination? A separate phase involving
> recordings of one individual with cerebral palsy is planned for a later
> semester and would be submitted as a protocol with a faculty PI after I
> complete the required training.
>
> Thank you, Jack ___

Keep the reply in `reports/irb/` (git-ignored) and cite it on the poster's
limitations panel: *"Evaluated on public de-identified speakers; recording
the intended user requires ethics approval and is the next phase."*


---

## Recording protocol

> **Future phase, not Fall 2026.** Recording the client or his family is
> gated on the university's human-subjects review (IRB training and an
> approved protocol). This semester's measurements use the public TORGO
> corpus and my voice for the live demo
> ([Fall 2026 plan, the public-corpus semester](#fall-2026-plan-the-public-corpus-semester)). This document
> is the plan for the client phase, kept current so that phase can start the
> day approval arrives.

This protocol creates data that can be compared across sessions and used for
personalized training.

### Before the first session

1. Obtain informed consent for recording, storage, model training, backups,
   repository access, and deletion requests.
2. Explain that voice recordings and derived embeddings are identifying data.
3. Fix the microphone position and room setup. The deployment microphone is
   the Movo USB lavalier, clipped at the collar (or, second choice, mounted on
   the chair near the shoulder), never at his face. The distance decides how
   much other voices intrude: [design.md](design.md#other-voices-in-the-room).
4. Choose one consistent family rater when a rater label will be used.
5. Select stop, sleep, and wake phrases with the speaker.
6. Run `python scripts/asr/check.py` and `python -m pytest`.
7. Make a test recording and verify level, clipping, playback, backup, and
   restore procedures.

### Session schedule

Prefer several 20–25 minute sessions across different days and times rather
than one long session. Stop early when fatigue or frustration rises. Record
both easier and harder days.

More days help more than more takes. In the EasyCall CPU pilot runs (31 dysarthric speakers,
six sessions each; [research/round-3.md](research/round-3.md) §7.2 -
context, not a project measurement) every additional enrolment day was worth
about five points of matcher accuracy with no plateau at five, and the most
recent day alone beat the oldest alone by eight. So record one pass over the
vocabulary in every session rather than many passes on one day.

Use the same microphone position and room treatment whenever possible, the
lavalier clipped at the same spot every session, since a capsule that moves is
a channel change the matcher cannot tell from a change in his voice. Record
any unavoidable change in the session metadata.

### Session order

1. Start the session.
2. Record the speaker's 0–10 speech and fatigue ratings.
3. Record the fixed rater label, condition, microphone, and room.
4. Run the ten-item probe once. The probe measures within-session drift; it
   is never used to calibrate the matcher (in the Round 3 CPU pilot runs, §6.4,
   a ten-word estimate of the day's shift was noisier than the shift itself).
5. Record critical and functional prompts -- the whole vocabulary, one take
   each, every session.
6. Continue with scheduler-selected phonetic prompts.
7. Record a few minutes of ordinary speech that is *not* in the vocabulary
   (conversation, picture description). This is what a future open-set
   rejection threshold would be measured on, and it is sentence-path
   training data in its own right.
8. Take breaks as needed; do not fill silence with coaching speech.
9. Run the same probe again.
10. Record end-of-session speech and fatigue ratings.
11. End the session, review pending takes, and create a backup.

### Elicitation

The operator may read or play a prompt before recording, but the prompt voice
must not overlap the target recording. Do not interrupt a slow response. Mark
partial, unsure, or noisy takes during review instead of correcting them to the
prompt automatically.

### Review

Review every take with headphones when practical. Enter what was actually said
and reject audio that cannot be transcribed reliably. A usable variation or
misreading is still useful when labeled correctly.

Live utterances appear in the queue marked **live** with the recogniser's guess
in the box. Listen before accepting: the guess is not a transcript, and the
server will not accept a live clip with an empty box. Accepting the guess
unchanged after listening is fine; accepting it unheard teaches the model its
own errors.

### Control phrases

Record stop, sleep, and wake phrases at least five times each and across more
than one session. Test false activations against ordinary conversation before
enabling live use.

### Other voices

Once the vocabulary is settled, have one or two family members record it
once (same microphone, their own session). Their takes never enter the bank;
they train the projection (in the Round 3 CPU pilot runs an in-vocabulary
projection was worth +18 points on EasyCall, §7.1). Re-consent covers this:
it is another person's voice.

### Data handling

Back up at the end of every session. Do not copy the corpus to personal cloud
storage or send audio through messaging applications. Revisit consent before
adding another person's voice or granting repository access.
