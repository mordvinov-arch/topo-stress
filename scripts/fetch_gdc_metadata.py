# GDC: загрузка технических метаданных (aliquot barcode -> plate_id, center).
# Для каждого file_id из sample sheet делается GET /files/{id}?expand=...,
# из штрихкода аликвоты извлекаются plate (предпоследний сегмент)
# и sequencing_center (последний сегмент) баркода TCGA.

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from topostress.config import DATA_DIR, PROCESSED_DATA_DIR  # noqa: E402

SHEET = os.path.join(DATA_DIR, "gdc", "gdc_sample_sheet.tsv")
OUT = os.path.join(PROCESSED_DATA_DIR, "gdc_metadata.csv")

FIELDS = "file_id,cases.samples.sample_id,cases.samples.portions.analytes.aliquots.submitter_id,cases.samples.portions.analytes.aliquots.aliquot_id"
EXPAND = "cases.samples.portions.analytes.aliquots"


def fetch_one(file_id):
    url = f"https://api.gdc.cancer.gov/files/{file_id}?fields={FIELDS}&expand={EXPAND}"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                d = r.json().get("data", {})
                alq = []
                for case in d.get("cases", []):
                    for sample in case.get("samples", []):
                        for portion in sample.get("portions", []):
                            for analyte in portion.get("analytes", []):
                                for a in analyte.get("aliquots", []):
                                    alq.append(a.get("submitter_id"))
                return file_id, alq
            else:
                time.sleep(2 * (attempt + 1))
        except Exception:
            time.sleep(2 * (attempt + 1))
    return file_id, []


def main():
    sheet = pd.read_csv(SHEET, sep="\t")
    file_ids = sheet["File ID"].tolist()
    print("files:", len(file_ids), flush=True)

    rows = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=12) as ex:
        for file_id, alq in ex.map(fetch_one, file_ids):
            rows[file_id] = alq
    print("fetched in %.0fs" % (time.time() - t0), flush=True)

    records = []
    missing = 0
    for _, row in sheet.iterrows():
        fid = row["File ID"]
        alq = rows.get(fid, [])
        if not alq:
            missing += 1
            plate, center, aliquot = None, None, None
        else:
            barcode = alq[0]
            parts = barcode.split("-")
            plate = parts[-2] if len(parts) >= 2 else None
            center = parts[-1] if len(parts) >= 1 else None
            aliquot = barcode
        records.append({
            "file_id": fid,
            "file_name": row["File Name"],
            "case_id": row["Case ID"],
            "sample_id": row["Sample ID"],
            "tissue_type": row["Tissue Type"],
            "tumor_descriptor": row["Tumor Descriptor"],
            "specimen_type": row["Specimen Type"],
            "preservation_method": row["Preservation Method"],
            "aliquot_barcode": aliquot,
            "plate_id": plate,
            "sequencing_center": center,
        })
    df = pd.DataFrame(records)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df.to_csv(OUT, index=False)
    print("saved:", OUT, flush=True)
    print("missing aliquots:", missing, flush=True)
    print("centers:", df["sequencing_center"].value_counts(dropna=False).to_dict(), flush=True)
    print("plates:", df["plate_id"].nunique(), "unique", flush=True)


if __name__ == "__main__":
    main()
