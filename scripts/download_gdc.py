# GDC: массовая загрузка файлов из манифеста через HTTPS API.
# Клиент gdc-client не требуется: данные качаются напрямую с
# https://api.gdc.cancer.gov/data/<file_id>. Открытые файлы — без токена,
# контролируемые — с X-Auth-Token (опция --token).

import argparse
import hashlib
import os
import sys
import threading
import time

import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from topostress.config import DATA_DIR  # noqa: E402

BASE_URL = "https://api.gdc.cancer.gov/data"
print_lock = threading.Lock()


def read_manifest(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            row = dict(zip(header, parts))
            row["md5"] = row.get("md5", "").strip()
            row["size"] = int(row.get("size", 0))
            rows.append(row)
    return rows


def md5_of(path, chunk=1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def already_done(row, out_dir):
    fp = os.path.join(out_dir, row["filename"])
    if not os.path.exists(fp) or os.path.getsize(fp) != row["size"]:
        return False
    return md5_of(fp) == row["md5"]


def download_one(row, out_dir, token, session, max_retries=5):
    fp = os.path.join(out_dir, row["filename"])
    tmp = fp + ".part"
    headers = {"X-Auth-Token": token} if token else {}
    last_err = None
    for attempt in range(max_retries):
        try:
            r = session.get(BASE_URL + "/" + row["id"], headers=headers, timeout=120)
            if r.status_code == 403:
                return "403", "требуется токен (контролируемый доступ)"
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
            if os.path.getsize(tmp) != row["size"]:
                raise IOError("размер не совпадает: %d != %d" % (os.path.getsize(tmp), row["size"]))
            if md5_of(tmp) != row["md5"]:
                raise IOError("MD5 не совпадает")
            os.replace(tmp, fp)
            return "ok", ""
        except Exception as e:
            last_err = e
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
            time.sleep(2 ** attempt)
    return "error", str(last_err)


def main():
    ap = argparse.ArgumentParser(description="GDC bulk download from manifest")
    ap.add_argument("manifest", nargs="?", default=os.path.join(DATA_DIR, "gdc", "gdc_manifest.txt"))
    ap.add_argument("--token", default=None, help="файл с auth-токеном GDC (или сам токен)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "gdc"))
    args = ap.parse_args()

    token = None
    if args.token:
        with open(args.token, encoding="utf-8") as f:
            token = f.read().strip()

    rows = read_manifest(args.manifest)
    print("manifest: %d files, total %.2f GB" % (len(rows), sum(r["size"] for r in rows) / 1e9), flush=True)
    os.makedirs(args.out, exist_ok=True)

    todo = [r for r in rows if not already_done(r, args.out)]
    done = len(rows) - len(todo)
    print("уже скачано и проверено: %d, осталось: %d" % (done, len(todo)), flush=True)

    if not todo:
        print("всё уже на месте")
        return

    session = requests.Session()
    results = {"ok": 0, "403": 0, "error": 0}
    failures = []
    counter = [done]
    total_bytes = [0]

    def work(r):
        status, msg = download_one(r, args.out, token, session)
        with print_lock:
            counter[0] += 1
            results[status] += 1
            if status == "ok":
                total_bytes[0] += r["size"]
                print("[%d/%d] ok   %s" % (counter[0], len(rows), r["filename"]), flush=True)
            else:
                failures.append((r["id"], r["filename"], msg))
                print("[%d/%d] %s %s  (%s)" % (counter[0], len(rows), status, r["filename"], msg), flush=True)

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for r in todo:
            ex.submit(work, r)

    print("==== Итог ====", flush=True)
    print("ok: %d, 403 (нужен токен): %d, ошибки: %d" % (results["ok"], results["403"], results["error"]), flush=True)
    print("скачано: %.2f GB" % (total_bytes[0] / 1e9), flush=True)
    if failures:
        print("не скачано:", flush=True)
        for fid, fname, msg in failures:
            print("  %s  %s  %s" % (fid, fname, msg), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
