# GDC2: мутационный статус EGFR/KRAS/TP53 для TCGA-LUAD (cBioPortal, study luad_tcga_gdc).
# Xena/GDC недоступны напрямую в некоторых сетях, поэтому берём WES-мутации из cBioPortal
# (та же GDC-когорта, barcodes TCGA-XX-XXXX-01). Сопоставление с нашей клиникой по case_id.
#
# Выход: data/processed/gdc2_mutations.csv (case_id, EGFR, KRAS, TP53 — 0/1) + кэш JSON.

import json
import os
import sys
import time
import urllib.request

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from topostress.config import PROCESSED_DATA_DIR  # noqa: E402

STUDY = "luad_tcga_gdc"
PROFILE = f"{STUDY}_mutations"
SAMPLE_LIST = f"{STUDY}_all"
GENES = {"EGFR": 1956, "KRAS": 3845, "TP53": 7157}
API = "https://www.cbioportal.org/api"
MERGED = os.path.join(PROCESSED_DATA_DIR, "gdc2_clinical_merged.csv")
OUT_CSV = os.path.join(PROCESSED_DATA_DIR, "gdc2_mutations.csv")
OUT_JSON = os.path.join(PROCESSED_DATA_DIR, "cbioportal_luad_gdc_mutations.json")


def fetch(url, retries=4, timeout=90):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json",
                                                       "User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            print(f"retry {attempt + 1} for {url}: {exc}", flush=True)
            time.sleep(3 * (attempt + 1))


def main():
    raw = {}
    for gene, entrez in GENES.items():
        url = (f"{API}/molecular-profiles/{PROFILE}/mutations"
               f"?sampleListId={SAMPLE_LIST}&molecularProfileId={PROFILE}"
               f"&entrezGeneId={entrez}")
        muts = fetch(url)
        raw[gene] = muts
        print(f"{gene}: {len(muts)} мутационных событий", flush=True)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=1)

    status = {}
    for gene, muts in raw.items():
        cases = set()
        for m in muts:
            sid = m.get("sampleId") or ""
            if len(sid) >= 12:
                cases.add(sid[:12].upper())
        status[gene] = cases

    merged = pd.read_csv(MERGED)
    cases = merged["case_id"].astype(str).str.upper().unique()
    rows = {"case_id": cases}
    for gene in GENES:
        rows[gene] = [1 if c in status[gene] else 0 for c in cases]
    mut = pd.DataFrame(rows)
    mut = mut.sort_values("case_id").reset_index(drop=True)

    mut.to_csv(OUT_CSV, index=False)

    print("всего кейсов в клинике:", len(cases), flush=True)
    for gene in GENES:
        n_mut = int(mut[gene].sum())
        n_found = len(status[gene])
        print(f"{gene}: мутаций по cBioPortal (кейсы) = {n_found}, в нашей когорте = {n_mut} "
              f"({n_mut / max(len(cases), 1) * 100:.1f}%)", flush=True)
    print("saved:", OUT_CSV, flush=True)


if __name__ == "__main__":
    main()
