# Agent Brief

Use this file as compressed context for a new Codex/Claude/AI coding session. Read root
`AGENTS.md` first for operating rules.

## Objective

Build the best open-weight Uzbek ASR model using `openai/whisper-large-v3`. Optimize
Uzbek WER/CER only. Multilingual forgetting is acceptable.

## Current Best

`partial_ft_usc_baseline`

- path: `archive/partial_ft_usc/model/`;
- USC 104.63h;
- encoder 0-23 frozen;
- encoder 24-31 + decoder trainable;
- WER `0.2005258480`;
- CER `0.0529079419`;
- immutable.

## Key Negative Result

One-epoch USC full FT:

- all 1.543B parameters;
- BF16;
- encoder LR `2e-6`;
- decoder LR `8e-6`;
- WER `0.2221522737`;
- CER `0.0565825834`.

Conclusion: do not assume full FT is better; preserve lower encoder acoustics unless
larger-data evidence says otherwise.

## Active Work

Autonomous LR/freeze search:

- tmux: `whisper_lr_search`;
- controller: `scripts/lr_search/autonomous_search.py`;
- logs: `reports/lr_search/autonomous_search*.log`;
- outputs: `outputs_lr_search/`;
- current phase at 2026-06-24 documentation rebuild: Phase 1A;
- `2e-6` decoder-only completed/stable;
- `8e-6` decoder-only reached step 300; final validation was running at the 08:54 UTC
  documentation verification.

Check actual state before acting:

```bash
tmux ls
pgrep -af 'autonomous_search.py|run_experiment.py|src/train.py'
tail -40 reports/lr_search/autonomous_search_console.log
nvidia-smi
```

Do not start a second controller.

## Data

Gold master:

- path: `data/gold_master/`;
- 207.1150h;
- USC + Common Voice Uzbek + FLEURS;
- 186.4037h train / 10.3556h validation / 10.3557h test;
- zero known path/hash/speaker leakage;
- FeruzaSpeech blocked by gated access.

Search proxies:

- `data/lr_search/coarse_10h/`;
- `data/lr_search/main_30h/`;
- 50/40/10 USC/CV/FLEURS duration mix.

Test is not loaded during search. Enforced config:

```yaml
load_test_split: false
evaluate_test_after_training: false
```

## Training Modes

- `decoder_only`: 906,521,600 trainable.
- `encoder_24_31_plus_decoder`: 1,063,930,880.
- `encoder_16_31_plus_decoder`: 1,221,340,160.
- full FT: 1,543,490,560.

Force `language="uz"`, `task="transcribe"`. New large-v3 runs use BF16, gradient
checkpointing, batch 1, gradient accumulation 32, gradient clipping 1.0 unless measured
evidence changes the plan.

## Search Decision Rules

- primary: validation WER;
- secondary: CER, convergence, stability, generalization;
- WER tie: delta `<0.003`;
- CER tie: delta `<0.001`;
- promote top two from coarse;
- never use test for ranking.

Final output required:

```text
reports/lr_search/FINAL_RECOMMENDATION.md
```

It must select decoder LR, encoder LR, freeze boundary, regime, confidence, and the
recommended 207h Gold config.

## Critical Constraints

- Never modify `archive/partial_ft_usc/`.
- Do not revert unrelated dirty-worktree changes.
- Do not train raw Silver/Bronze.
- Do not launch four epochs without explicit new evidence/approval.
- Do not bypass checkpoint security protections.
- Back up important files before risky edits.
- Update docs and `state.json` after meaningful changes.

## Navigation

- live state: `docs/STATUS.md`;
- architecture: `docs/ARCHITECTURE.md`;
- data: `docs/DATA_GOVERNANCE.md`;
- training/search: `docs/TRAINING_AND_SEARCH.md`;
- operations: `docs/OPERATIONS_RUNBOOK.md`;
- history: `docs/EXPERIMENT_LEDGER.md`;
- decisions/failures: `docs/DECISION_LOG.md`, `docs/FAILURE_LOG.md`;
- next work: `docs/ROADMAP.md`;
- recovery: `docs/DISASTER_RECOVERY.md`.
