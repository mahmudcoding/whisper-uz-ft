# LR Search Data Leakage Audit

- Overall status: **PASS**
- Training source: `train.csv` only.
- Checkpoint/model selection source: `val.csv` only.
- Test policy: `test.csv` is neither loaded nor evaluated during LR search.

## Exact Paths and Hash Guards

### `/home/mahmud/whisper-uz-ft/data/lr_search/coarse_10h`

- train: `/home/mahmud/whisper-uz-ft/data/lr_search/coarse_10h/train.csv`; rows `8733`; SHA-256 `70b85fd90c759bf90cfff6f0a20fe65e51c870295b2335e9bd8b2e33d84f0af8`
- val: `/home/mahmud/whisper-uz-ft/data/lr_search/coarse_10h/val.csv`; rows `845`; SHA-256 `11a78da6809273d893a875d764312908d4eb0f8b762bd5c09b5a626799fbc197`
- test: `/home/mahmud/whisper-uz-ft/data/lr_search/coarse_10h/test.csv`; rows `818`; SHA-256 `bb08474ed6dd891d4703eb9eaf500378eacefd5bfe4e4fb2731edcbb6a514859`
- Exact path overlap: `{'train_val': 0, 'train_test': 0, 'val_test': 0}`
- Transcript collisions: `{'train_val': 36, 'train_test': 27, 'val_test': 19}`
- Status: **PASS**

- Warning: 36 normalized transcript strings occur in both train and val; audio and reliable speakers remain disjoint
- Warning: 27 normalized transcript strings occur in both train and test; audio and reliable speakers remain disjoint
- Warning: 19 normalized transcript strings occur in both val and test; audio and reliable speakers remain disjoint

### `/home/mahmud/whisper-uz-ft/data/lr_search/main_30h`

- train: `/home/mahmud/whisper-uz-ft/data/lr_search/main_30h/train.csv`; rows `26249`; SHA-256 `3df230e327c6672cfe1d86c6970157c49518d8dfcecb6f2b217ac8a4020bfc84`
- val: `/home/mahmud/whisper-uz-ft/data/lr_search/main_30h/val.csv`; rows `847`; SHA-256 `705c6eb3a111c44c526d1a4b9be0b8895ff7bc6ccb946e751278d034ef75fdca`
- test: `/home/mahmud/whisper-uz-ft/data/lr_search/main_30h/test.csv`; rows `816`; SHA-256 `b40865d4320ede6db1fe27d937f1adaa6b5d201d4581b2d548aed5399cee29c8`
- Exact path overlap: `{'train_val': 0, 'train_test': 0, 'val_test': 0}`
- Transcript collisions: `{'train_val': 99, 'train_test': 75, 'val_test': 13}`
- Status: **PASS**

- Warning: 99 normalized transcript strings occur in both train and val; audio and reliable speakers remain disjoint
- Warning: 75 normalized transcript strings occur in both train and test; audio and reliable speakers remain disjoint
- Warning: 13 normalized transcript strings occur in both val and test; audio and reliable speakers remain disjoint

## Pipeline Verification

- Code checks: `{'trainer_train_dataset_is_train': True, 'trainer_eval_dataset_is_validation': True, 'test_loading_is_configurable': True, 'test_evaluation_is_guarded': True}`
- Config errors: `[]`

## Fix Applied

Before this audit, LR-search runs disabled final test evaluation but still loaded and
feature-preprocessed `test.csv`. The training loader now supports `load_test_split: false`,
all LR-search configs enforce it, the runner rejects any config that loads/evaluates test,
and the comparison tool rejects metrics containing test results.

Repeated transcript text across splits is not treated as leakage when audio paths and
reliable speaker identities are disjoint: read-speech corpora legitimately contain common
short phrases. Test manifest hashes above are the immutable guards for the search.
