# GEO: независимая когорта LUAD GSE31210 (GPL570, U133A) для валидации физиотипов.
# Скачивает series matrix + аннотацию платформы, строит:
#   data/geo/GSE31210_series_matrix.txt.gz   (сырой)
#   data/geo/GPL570.annot.gz                 (сырой)
#   data/processed/gse31210_expression.csv   (образцы x гены, log2, из нашей HVG-сигнатуры)
#   data/processed/gse31210_probe_map.csv    (зонд -> ген)
#   data/processed/gse31210_clinical.csv     (GSM, os_days, event, stage, ...)
#
# Скачивание устойчиво к недоступности системного DNS: при сбое резолвит хост
# через DoH (dns.google / cloudflare) и качает через curl --resolve.

import gzip
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from topostress.config import DATA_DIR, PROCESSED_DATA_DIR  # noqa: E402

GEO_DIR = os.path.join(DATA_DIR, "geo")
SERIES_URL = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE31nnn/GSE31210/"
              "matrix/GSE31210_series_matrix.txt.gz")
ANNOT_URL = "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz"

RAW_SERIES = os.path.join(GEO_DIR, "GSE31210_series_matrix.txt.gz")
RAW_ANNOT = os.path.join(GEO_DIR, "GPL570.annot.gz")
OUT_EXPR = os.path.join(PROCESSED_DATA_DIR, "gse31210_expression.csv")
OUT_MAP = os.path.join(PROCESSED_DATA_DIR, "gse31210_probe_map.csv")
OUT_CLIN = os.path.join(PROCESSED_DATA_DIR, "gse31210_clinical.csv")

CURL = shutil.which("curl")


def resolve_ip(host):
    try:
        return socket.gethostbyname(host)
    except OSError:
        pass
    for doh in ("https://dns.google/resolve", "https://cloudflare-dns.com/dns-query"):
        try:
            req = urllib.request.Request(f"{doh}?name={host}&type=A",
                                         headers={"accept": "application/dns-json",
                                                  "User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                d = json.loads(r.read())
            for a in d.get("Answer", []):
                if a.get("type") == 1:
                    return a["data"]
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError(f"не удалось разрешить {host}")


def download(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as r:
            with open(dest, "wb") as f:
                shutil.copyfileobj(r, f)
        print("downloaded (urllib):", dest, flush=True)
        return
    except Exception as exc:  # noqa: BLE001
        print(f"urllib failed ({exc}); fallback via DoH+curl", flush=True)
    if not CURL:
        raise RuntimeError("curl недоступен, а urllib не смог скачать")
    host = urllib.parse.urlparse(url).netloc
    ip = resolve_ip(host)
    cmd = [CURL, "--resolve", f"{host}:443:{ip}", "-sS", "-L", "-o", dest, url]
    subprocess.run(cmd, check=True)
    print(f"downloaded (curl --resolve {host}->{ip}):", dest, flush=True)


def parse_series(path):
    # GSE31210 series matrix: метаданные образцов идут строками !Sample_* с
    # табуляцией (одна колонка на образец), затем таблица экспрессии.
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    meta = {}
    data_start = None
    for i, line in enumerate(lines):
        s = line.rstrip("\n")
        if s.startswith("!") and "\t" in s:
            parts = s.split("\t")
            meta.setdefault(parts[0], []).append([p.strip().strip('"') for p in parts[1:]])
        if s.startswith("!series_matrix_table_begin"):
            data_start = i + 1
            break

    gsm = meta["!Sample_geo_accession"][0]
    n_samples = len(gsm)
    print("samples in series:", n_samples, flush=True)

    header = [h.strip().strip('"') for h in lines[data_start].rstrip("\n").split("\t")]
    probe_ids = []
    rows = []
    for line in lines[data_start + 1:]:
        s = line.rstrip("\n")
        if s.startswith("!series_matrix_table_end"):
            break
        parts = s.split("\t")
        probe_ids.append(parts[0].strip('"'))
        rows.append([float(x) if x not in ("", "NA", "null") else np.nan for x in parts[1:]])
    expr = pd.DataFrame(np.array(rows, dtype=float), index=probe_ids, columns=header[1:])
    expr.index.name = "probe"
    print("expression:", expr.shape, flush=True)

    # Характеристики в GSE31210 смещены для части образцов (ячейки несут своё
    # имя), поэтому парсим каждую ячейку независимо: "name: value".
    clin_rows = []
    char_rows = meta.get("!Sample_characteristics_ch1", [])
    for s in range(n_samples):
        rec = {}
        for vals in char_rows:
            v = vals[s] if s < len(vals) else ""
            if ": " in v:
                n, _, val = v.partition(": ")
                rec.setdefault(n.strip(), val.strip())
        rec["gsm"] = gsm[s]
        clin_rows.append(rec)
    clin = pd.DataFrame(clin_rows)
    clin["tissue_type"] = meta.get("!Sample_source_name_ch1", [[""] * n_samples])[0]
    clin["title"] = meta.get("!Sample_title", [[""] * n_samples])[0]
    return expr, clin


def parse_annot(path):
    probes = {}
    in_table = False
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.rstrip("\n")
            if s.startswith("!platform_table_begin"):
                in_table = True
                continue
            if s.startswith("!platform_table_end"):
                break
            if not in_table:
                continue
            parts = s.split("\t")
            if len(parts) < 3:
                continue
            probe = parts[0]
            if probe == "ID":
                continue
            symbol = parts[2].strip()
            probes[probe] = symbol
    return probes


def main():
    os.makedirs(GEO_DIR, exist_ok=True)
    if not os.path.exists(RAW_SERIES):
        download(SERIES_URL, RAW_SERIES)
    if not os.path.exists(RAW_ANNOT):
        download(ANNOT_URL, RAW_ANNOT)

    expr, clin = parse_series(RAW_SERIES)
    probe_map = parse_annot(RAW_ANNOT)
    print("probes annotated:", len(probe_map), flush=True)

    map_df = pd.DataFrame({"probe": list(probe_map), "symbol": list(probe_map.values())})
    map_df = map_df[map_df["symbol"] != ""].drop_duplicates("probe")
    map_df.to_csv(OUT_MAP, index=False)

    wide = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "gdc_wide.csv"), index_col=0)
    hvgs = set(wide.columns)

    gene_probes = map_df[map_df["symbol"].isin(hvgs)]
    common_probes = list(set(gene_probes["probe"]) & set(expr.index))
    print("HVG-genes covered by probes:", gene_probes["symbol"].nunique(),
          "| probes:", len(common_probes), flush=True)

    expr_sub = expr.loc[common_probes]
    expr_sub = expr_sub.assign(symbol=gene_probes.set_index("probe").loc[common_probes, "symbol"].values)

    # несколько зондов на ген -> среднее
    g_expr = expr_sub.groupby("symbol").mean()
    # порядок колонок как в gdc_wide (топ-500 HVG по дисперсии)
    order = [g for g in wide.columns if g in g_expr.index]
    g_expr = g_expr.reindex(order)
    g_expr.to_csv(OUT_EXPR)
    print("gene expression:", g_expr.shape, "->", OUT_EXPR, flush=True)

    clin = clin[clin["gsm"].isin(expr.columns)].reset_index(drop=True)
    # Нормализация клиники в единую схему для geo_validate.py
    out = pd.DataFrame()
    out["gsm"] = clin["gsm"]
    out["title"] = clin["title"]
    out["tissue"] = np.where(clin["tissue"].fillna("").str.lower().str.contains("tumor"),
                             "Tumor", "Normal")
    out["stage"] = clin.get("pathological stage", clin.get("pstage iorii", pd.Series(index=clin.index)))
    os_days = pd.to_numeric(clin.get("days before death/censor"), errors="coerce")
    death = clin.get("death", "").fillna("")
    out["os_days"] = os_days
    out["event"] = np.where(death.str.lower().isin(["dead", "died", "1"]), 1, 0)
    relapse = clin.get("relapse", "").fillna("")
    out["relapse_event"] = np.where(relapse.str.lower() == "relapsed", 1, 0)
    out["rfs_days"] = pd.to_numeric(clin.get("days before relapse/censor"), errors="coerce")
    excl = clin.get("exclude for prognosis analysis due to incomplete resection or adjuvant therapy",
                    "").fillna("")
    out["exclude"] = np.where(excl.str.lower() == "exclude", 1, 0)
    out["age_years"] = pd.to_numeric(clin.get("age (years)"), errors="coerce")
    out["gender"] = clin.get("gender", "")
    out.to_csv(OUT_CLIN, index=False)
    print("clinical rows:", out.shape, "->", OUT_CLIN, flush=True)
    print("tissue counts:", out["tissue"].value_counts().to_dict(), flush=True)
    print("events:", int(out["event"].sum()), "of", len(out), flush=True)
    print("clinical columns:", list(out.columns), flush=True)


if __name__ == "__main__":
    main()
