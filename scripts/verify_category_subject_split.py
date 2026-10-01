"""Check that the OakInk-Image category x subject sweep measured what its claim assumes.

That sweep now carries "necessity is refuted", and its bundle uses a workaround: labels are packed as
``<fine>@<left>@-@<right>`` and parsed with the ``oakink2_scene_verb`` mode, because ``coarsen_labels`` has no
generic OakInk-Image mode. This script rebuilds the split for several seeds with the sweep's own recorded arguments,
calling the same functions in the same way ``main()`` does, and asserts:

1. the packed factors agree with the parsed cell, and the cells are (category, subject) pairs;
2. every target trajectory belongs to a held-out cell;
3. the naive arm's training set contains no trajectory of any held-out cell;
4. the informed arm's training set does contain them;
5. the sweep's own ``assert_split_sound`` reports zero fine-composition leak on both arms.

It trains nothing. Exit code 1 on the first failed assertion.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import experiment_paired_composition as E  # noqa: E402
from caredex.data.base import TrajectoryBundle  # noqa: E402

transitions_of = getattr(E, "transitions_of", None)
if transitions_of is None:  # it lives in the older experiment module
    from experiment_data_efficiency import transitions_of  # noqa: E402

RUN = ROOT / "runs" / "oakink_category_subject"
args = json.loads((RUN / "results.json").read_text(encoding="utf-8"))["args"]
print("sweep args:", {k: args[k] for k in ("bundle", "granularity", "held_compositions", "min_chains",
                                           "min_per_composition", "budgets", "stride")})

bundle = TrajectoryBundle.load(str(ROOT / args["bundle"]))
raw = list(bundle.labels)
fine = list(raw)
bundle.labels = E.coarsen_labels(raw, args["granularity"])
coarse = list(bundle.labels)
print("example raw label  :", raw[0])
print("example coarse cell:", coarse[0])
print("example cell parsed:", transitions_of(coarse[0]))

# 1. the packed factors must agree with the parsed cell, and the right factor must be the subject in the fine label.
for r, c in zip(raw, coarse):
    body, _, tail = r.partition("@")
    parts = [p for p in tail.split("@") if p not in ("", "-")]
    assert len(parts) == 2, f"unexpected packing: {r}"
    left, right = parts
    assert right == body.split("subject=")[1], f"right factor is not the subject: {r}"
    assert left in c and right in c, f"parsed cell {c} does not carry the packed factors {left}, {right}"
categories = {c.split("->")[0] for c in coarse}
subjects = {c.split("->")[1] for c in coarse}
print(f"labels checked {len(raw)}: {len(set(coarse))} cells, {len(categories)} categories, {len(subjects)} subjects")

budget = args["budgets"][0]
for seed in (0, 1, 7, 13, 25, 39):
    split = E.build_paired_split(
        bundle, args["held_compositions"], seed, fine_labels=fine, min_chains=args["min_chains"],
    )
    naive_idx, informed_idx = E.sample_pools(split, budget, seed, bundle.labels, args["min_per_composition"])
    sound = E.assert_split_sound(split, naive_idx, informed_idx, fine, bundle.labels)
    held = set(split["held_compositions"])

    def cells(idx):
        return {f"{a}->{b}" for i in idx for a, b in transitions_of(coarse[i])}

    t_cells, n_cells, i_cells = cells(split["target"]), cells(naive_idx), cells(informed_idx)
    assert t_cells <= held, f"seed {seed}: target outside held cells: {sorted(t_cells - held)[:3]}"
    assert not (n_cells & held), f"seed {seed}: naive arm saw held cells {sorted(n_cells & held)[:3]}"
    assert i_cells & held, f"seed {seed}: informed arm saw no held cell"
    assert sound["leak_naive"] == 0 and sound["leak_informed"] == 0, f"seed {seed}: leak {sound}"
    print(f"seed {seed:2d}: held {sound['held']}, informed covered {sound['covered']}, "
          f"target {len(split['target'])} traj, naive {len(naive_idx)}, informed {len(informed_idx)}, "
          f"leak {sound['leak_naive']}/{sound['leak_informed']}, example held {sorted(held)[0]}")

print("PASS: cells are category x subject pairs, targets lie in held cells, the naive arm saw none of them, "
      "the informed arm did, and neither arm leaks a target's fine composition.")
