# Supplementary ZIP for the TMLR submission: build and anonymisation report

Built by `<scratch>/tmlr/build_supplement.py` on 2026-10-05. Staging tree `<scratch>/tmlr/supplement/`, archive `<scratch>/tmlr/supplement.zip`.

## Archive

- Size: 1,407,806 bytes (1.41 MB); TMLR limit 100 MB: OK.
- Files: 432 (staging tree, including README.md and scrub_log.txt).
- sweeps/: 80 sweep directories, 85 files (results.json + sidecars).
- gates/: 146 files. preregistration/: 21 files. readings/: 2 files.
- analysis/: 19 files (18 from sync_release.py's ANALYSIS list, plus ['cell_random_effects.json'] because scripts/verify_tmlr_draft.py reads it). None missing.
- code/: 41 package files under src/caredex/, 114 scripts, pyproject.toml. Excluded scripts (23): build_tmlr_latex.py, experiment_robot_transfer.py, lineC_chain.py, lineC_q10_followup.py, lineC_q10_pipeline.py, lineC_q9_pipeline.py, make_web_video_label_alignment.py, render_lineC_video.py, rl_arms_summary.py, rl_composition_sweep.py, rl_paired_sweep.py, rl_smoke.py, rl_stage_breakdown.py, rl_stageF_primary.py, rl_stageG_primary.py, rl_stageH_primary.py, rl_stageI_primary.py, rl_vec_smoke.py, sync_release.py, tmlr_md_to_docx.before_refcheck.py, tmlr_md_to_docx.py, train_adroit.py, train_rl_policy.py.
- video/label_alignment.mp4: 354,861 bytes.

## Anonymisation

- Text files scanned: 430. Files changed: 48. Replacements: 96. Full line-by-line log: `scrub_log.txt` in the archive root.
- Replacements by rule: repo-path 89, full-name 2, institution-full 2, handle 2, home-path 1.
- Files changed (48):
  - analysis/constituent_leak_pair.json
  - code/scripts/build_docx.py
  - code/scripts/build_latex.py
  - code/scripts/experiment_dexpilot_transfer.py
  - gates/composition_leak_oakink_category_subject.txt
  - gates/composition_leak_oakink_category_subject_h3.txt
  - gates/composition_leak_oakink_category_subject_h4.txt
  - gates/composition_leak_oakink_class_attr.txt
  - gates/composition_leak_oakink_class_attr2.txt
  - gates/composition_leak_oakink_class_attr2_h3.txt
  - gates/composition_leak_oakink_class_attr2_h4.txt
  - gates/composition_leak_oakink_class_attr_h3.txt
  - gates/composition_leak_oakink_class_attr_h4.txt
  - gates/composition_leak_oakink_class_date.txt
  - gates/composition_leak_oakink_class_date_h3.txt
  - gates/composition_leak_oakink_class_date_h4.txt
  - gates/composition_leak_oakink_object_date.txt
  - gates/composition_leak_oakink_object_date_h3.txt
  - gates/composition_leak_oakink_object_date_h4.txt
  - gates/composition_leak_oakink_object_intent.txt
  - gates/composition_leak_oakink_object_intent_h3.txt
  - gates/composition_leak_oakink_object_intent_h4.txt
  - gates/composition_leak_oakink_object_subject.txt
  - gates/composition_leak_oakink_object_subject_h3.txt
  - gates/composition_leak_oakink_object_subject_h4.txt
  - gates/composition_leak_taco_action_tool_h3.txt
  - gates/cover_oakink_category_subject.txt
  - gates/cover_oakink_category_subject_h3.txt
  - gates/cover_oakink_category_subject_h4.txt
  - gates/cover_oakink_class_attr.txt
  - gates/cover_oakink_class_attr2.txt
  - gates/cover_oakink_class_attr2_h3.txt
  - gates/cover_oakink_class_attr2_h4.txt
  - gates/cover_oakink_class_attr_h3.txt
  - gates/cover_oakink_class_attr_h4.txt
  - gates/cover_oakink_class_date.txt
  - gates/cover_oakink_class_date_h3.txt
  - gates/cover_oakink_class_date_h4.txt
  - gates/cover_oakink_object_date.txt
  - gates/cover_oakink_object_date_h3.txt
  - gates/cover_oakink_object_date_h4.txt
  - gates/cover_oakink_object_intent.txt
  - gates/cover_oakink_object_intent_h3.txt
  - gates/cover_oakink_object_intent_h4.txt
  - gates/cover_oakink_object_subject.txt
  - gates/cover_oakink_object_subject_h3.txt
  - gates/cover_oakink_object_subject_h4.txt
  - gates/cover_taco_action_tool_h3.txt

### Final re-scan (asserted zero)

Residual hits of identifying strings after scrubbing: **0**.
- `maurice`: 0
- `mu-hua`: 0
- `muhua`: 0
- `wang mu`: 0
- `wang mu hua`: 0
- `nycu`: 0
- `yang ming`: 0
- `chiao tung`: 0
- `mauricewang`: 0
- `may.be13`: 0
- `github.com/maurice`: 0
- `hand-composition-axes`: 0
- `c:\users`: 0
- `c:/users`: 0
- `desktop\hand_ik`: 0
- `paper彙整`: 0
- `paper_fix`: 0
- `nctu` as a whole word: 0
- `nctu` as a substring of another word (not identifying, e.g. 'punctuation'): 0 lines

### Left in place, for the authors to decide

- `D:\datasets` (dataset drive path, not identifying): 30 occurrences, left as is.
- `caredex` (package name, case-insensitive): 302 occurrences in 102 files; `CareDex` in title case: 3 occurrences. Left as is; the README states it is a working name. The authors must decide whether the name is identifying (it is also the git user name of the private repository, but no git metadata is included).

### Residual notes

- The manuscript source (`docs/tmlr/PAPER_TMLR.md`) is not in the archive, so `verify_tmlr_draft.py` cannot run to completion from the archive alone; the README says so and names the per-quantity scripts that do.
- `verify_tmlr_draft.py` also reads the bundles for descriptive counts; the README explains they are rebuilt from the public datasets.
- Scripts carry the placeholders `<repo>`, `<scratch>`, `<home>`, `<external>` where absolute Windows paths were; any script that hard-coded such a path will need that line edited before it runs. See scrub_log.txt for which.
- Comments and docstrings in the code that mention the author in the third person were replaced by `<author>`; they read oddly but are not identifying.
- The `.ps1`/`.cmd` queue scripts in scripts/ were not included (the brief asked for scripts/*.py).
- Nothing in the repository was modified; only this report was written into docs/tmlr/final_check/.
