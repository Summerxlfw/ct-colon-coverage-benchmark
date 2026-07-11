#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split lockdown — 按病人切 HQColon usable 几何 (train/cal/test/holdout)。
单元=病人 (supine/prone 同侧, 防泄漏)。固定 seed, 落 JSON + sha256, 不再改。"""
import csv, json, hashlib, os, random

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "results", "coverage_labels.csv")
META = os.environ.get("HQCOLON_DIR", "./hqcolon_source") + "/meta-data.json"
OUT = os.path.join(HERE, "patient_split.json")
SEED = 20260612

# usable 池: centerline_ok & watertight & largest_frac>=0.9 (保守, 干净)
rows = list(csv.DictReader(open(CSV)))
usable = [r["case"] for r in rows
          if r.get("centerline_ok") == "True" and r.get("watertight") == "True"
          and float(r.get("lumen_largest_frac", 0)) >= 0.90]
# case(colon_NNN) → subject_id
recs = [json.loads(l) for l in open(META) if l.strip()]
case2sub = {r["nnunet_label_file"].replace(".mha", ""): r["subject_id"] for r in recs}
pat2cases = {}
for c in usable:
    s = case2sub.get(c)
    if s:
        pat2cases.setdefault(s, []).append(c)
patients = sorted(pat2cases)
rng = random.Random(SEED)
rng.shuffle(patients)

n = len(patients)
# holdout 12% / test 18% / cal 22% (≥50) / train 48%
n_hold = round(0.12 * n); n_test = round(0.18 * n); n_cal = round(0.22 * n)
holdout = patients[:n_hold]
test = patients[n_hold:n_hold + n_test]
cal = patients[n_hold + n_test:n_hold + n_test + n_cal]
train = patients[n_hold + n_test + n_cal:]
# pilot 子集 (~80 病人): 从各 split 取前若干, 保持与全量一致 (pilot⊂full)
pilot = dict(train=train[:48], cal=cal[:18], test=test[:14])

split = dict(seed=SEED, unit="patient", pool_criterion="centerline_ok & watertight & largest_frac>=0.9",
             n_patients=n, n_usable_scans=len(usable),
             train=train, cal=cal, test=test, holdout=holdout,
             counts=dict(train=len(train), cal=len(cal), test=len(test), holdout=len(holdout)),
             pilot=pilot, pilot_counts={k: len(v) for k, v in pilot.items()},
             patient_to_cases={p: pat2cases[p] for p in patients})
blob = json.dumps(split, sort_keys=True).encode()
split["sha256"] = hashlib.sha256(blob).hexdigest()
json.dump(split, open(OUT, "w"), indent=1)

print(f"usable scans={len(usable)} patients={n}")
print(f"split: train={len(train)} cal={len(cal)} test={len(test)} holdout={len(holdout)}  (cal>=50: {len(cal)>=50})")
print(f"pilot: {split['pilot_counts']}")
print(f"sha256={split['sha256'][:16]}...  → {OUT}")
