# Единый воспроизводимый конвейер.
# Запускает все скрипты анализа. Кэш: скрипт пропускается, если его артефакт
# уже есть (FORCE=1 — перезапустить всё; MAST_PERM / MAST_DRAWS — параметры).

import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from topostress.config import PROJECT_ROOT, FORCE

SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")


def a(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


STEPS = [
    ("data", "run_data.py", [a("data", "processed", "mast_wide.csv")]),
    ("repro_mmash", "run_repro_mmash.py", [a("results", "repro_mmash_done.flag")]),
    ("bayesian", "run_bayesian.py", [a("results", "bayesian_mast.json")]),
    ("evt", "run_evt.py", [a("results", "evt_mast.json")]),
    ("fda", "run_fda.py", [a("results", "fda_mast.json")]),
    ("rmt", "run_rmt.py", [a("results", "rmt_mast.json")]),
    ("hsic", "run_hsic.py", [a("results", "hsic_mast.json")]),
    ("conformal", "run_conformal.py", [a("results", "conformal_mast.json")]),
    ("info_geometry", "run_info_geometry.py", [a("results", "info_geometry_mast.json")]),
    ("tda", "run_tda.py", [a("results", "tda_mast.json")]),
    ("novelty", "run_novelty.py", [a("results", "novelty_mast.json")]),
    ("export_summaries", "export_summaries.py", [a("results", "bayesian_summary.csv")]),
    ("article_docx", "build_article_docx.py", [a("article", "mast_article.docx")]),
]


def done_check(artifacts):
    return all(os.path.exists(a) for a in artifacts)


def main():
    print("==== topo-stress: MAST analysis pipeline ====")
    for name, script, artifacts in STEPS:
        if not FORCE and done_check(artifacts):
            print(f"[skip] {name}: артефакты уже есть (FORCE=1 для перезапуска)")
            continue
        print(f"[run ] {name}: {script}", flush=True)
        s = time.time()
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        r = subprocess.run([sys.executable, os.path.join(SCRIPTS_DIR, script)], cwd=PROJECT_ROOT, env=env)
        if r.returncode != 0:
            print(f"[FAIL] {name}: код возврата {r.returncode}")
            sys.exit(1)
        print(f"[done] {name}: {time.time() - s:.0f} s", flush=True)
    print("==== Конвейер завершён ====")


if __name__ == "__main__":
    main()
