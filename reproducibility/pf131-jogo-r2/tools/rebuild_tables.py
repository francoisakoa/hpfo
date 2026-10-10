#!/usr/bin/env python3
"""Rebuild component tables from archived JSONs; do not run optimization."""
import argparse, csv, hashlib, json, math, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ABL = ("full","no_ipf","no_sharing","no_novelty","no_kicks",
       "no_island_exchange","no_local","no_adaptive_annealing","pf_local_only","pf_only")
GRIDS = {7: ABL, 8: ("np_100","np_200","np_300","np_400","np_500"),
         9: ("beta_1e-5","beta_1e-4","beta_1e-3","beta_1e-2"),
         10: ("alpha_0.95","alpha_0.98","alpha_0.99","alpha_1.00")}
ALIASES = {"beta_1e-4":"full","alpha_0.98":"full"}

def main(data, output):
    records = json.loads((ROOT/"provenance/RUN_INDEX.json").read_text())
    observed = {}
    for rec in records:
        path = data/rec["file"]
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != rec["sha256"]:
            raise RuntimeError("Hash mismatch: "+str(path))
        j = json.loads(raw)
        if j["objective_evaluations"] != j["objective_pf"]+j["objective_local"]:
            raise RuntimeError("Counter mismatch: "+str(path))
        if rec["dataset"] != "study":
            continue
        o = j["options"]
        success = j["objective"] <= o["target"]+o["target_tolerance"]
        if success != j["target_reached"]:
            raise RuntimeError("Success mismatch: "+str(path))
        key = (rec["configuration"], rec["atoms"], rec["run"])
        if key in observed:
            raise RuntimeError("Duplicate study result")
        observed[key] = {
            "success": int(success), "seconds": j["seconds"],
            "evaluations": j["objective_evaluations"], "objective": j["objective"],
            "iterations": j["iterations"],
        }
    if len(observed) != 820:
        raise RuntimeError("Expected 820 study results")
    output.mkdir(parents=True, exist_ok=False)
    for table, configs in GRIDS.items():
        rows = []
        repetitions = 30 if table == 7 else 10
        for cfg in configs:
            actual = ALIASES.get(cfg, cfg)
            for n in (75,102):
                group = [observed[(actual,n,r)] for r in range(1,repetitions+1)]
                row = {"configuration": cfg, "atoms": n, "runs": repetitions,
                       "successes": sum(x["success"] for x in group)}
                row["success_percent"] = 100*row["successes"]/repetitions
                for field in ("seconds","evaluations","objective","iterations"):
                    values = [x[field] for x in group]
                    if not all(math.isfinite(x) for x in values):
                        raise RuntimeError("Nonfinite aggregate input")
                    row["mean_"+field] = statistics.mean(values)
                    row["median_"+field] = statistics.median(values)
                    row["sample_sd_"+field] = statistics.stdev(values)
                rows.append(row)
        with (output/f"TABLE_{table}.csv").open("w",newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        if table == 7:
            lookup = {(x["configuration"],x["atoms"]):x for x in rows}
            lines = ["# Table 7 — completed component study",
                     "", "|Configuration|LJ75|LJ102|", "|---|---:|---:|"]
            for cfg in configs:
                cells = []
                for n in (75,102):
                    x = lookup[(cfg,n)]
                    cells.append(f'{x["success_percent"]:.1f}% ({x["successes"]}/30)')
                lines.append("|"+cfg+"|"+"|".join(cells)+"|")
            lines += ["", "Success is target attainment. Cost statistics include unsuccessful runs.",
                      "Sensitivity reference rows reuse full runs 1–10 only."]
            (output/"TABLE_7.md").write_text("\n".join(lines)+"\n")
    print("RAW_JSON_CHECKS=OK; STUDY=820; TABLE_7=600; SENSITIVITY_REFERENCES=RUNS_1_TO_10")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    main(args.data,args.output)
