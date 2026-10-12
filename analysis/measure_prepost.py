"""전처리 전 (자동 ROI) · 후 (최종) — 후보 수 · 후보 면적 · 처리 시간. 검출기 설정은 같음 (p2bd)."""
import sys, json, csv, time, random
from pathlib import Path
ROOT = Path("/Users/sj/devel_sj/computer-vision/CV")
SP = Path("/private/tmp/claude-501/-Users-sj-devel-sj-computer-vision/3a67ef65-f2e1-4551-8fc7-4e186d0342d7/scratchpad")
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(ROOT / "analysis"))
import run_road_pipeline as R
import cv2, numpy as np
import detect as TOPD, preprocess as TOPP, run_pipeline as RP
rcA = json.load(open(ROOT / "outputs/pipeline/run_20261008_005056/run_config.json"))
cfgA = RP.detector_cfg("D1hv", rcA["valley_ratio"], None, None, None, None, rcA["line_hi_abs"], rcA["overrides"])
cfg_a = TOPP.validate_config({"roi": "auto"})
rcF = json.load(open(ROOT / "outputs/pipeline/run_20261011_072601/run_config.json"))
pre_cfg, det = R.load_config(R.PRE_CFG), R.load_detection_config(R.DET_CFG)
cfgF = {**R.DEFAULT_CFG, **det["detect_cfg"], "line_hi_abs": rcF["line_hi_abs"]}
group = {r["image"]: r["group"] for r in csv.DictReader(open(R.SPLIT_CSV, encoding="utf-8"))}
name, _, images = R.list_images("rdd_dev")
import os
images = images[:int(os.environ.get("LIMIT", "0")) or None]
timed = set(p.name for p in random.Random(20261012).sample(images, min(100, len(images))))
out = []
SIFT = cv2.SIFT_create()
for i, path in enumerate(images, 1):
    row = {"image": path.name, "timed": path.name in timed}
    # 전처리 전 (A)
    t0 = time.perf_counter()
    img = R.imread(path)
    ref, geo = TOPP.geometry_preprocess(img, cfg_a, path.name)
    t1 = time.perf_counter()
    fixed, _ = TOPP.preprocess_condition(ref, cfg_a, "none", set(), {})
    t2 = time.perf_counter()
    dA = [d for d in TOPD.detect(fixed, cfgA) if d["type"] == "crack"]
    t3 = time.perf_counter()
    s = geo["scale_x"] * geo["scale_y"]
    gA = cv2.cvtColor(fixed, cv2.COLOR_BGR2GRAY)
    row.update(A_edges=int((cv2.Canny(gA, 100, 200) > 0).sum()), A_kp=len(SIFT.detect(gA, None)), A_px=int(gA.size))
    row.update(A_n=len(dA), A_area=sum(d["area"] for d in dA) / s, A_box=sum(d["bbox"][2] * d["bbox"][3] for d in dA) / s, A_pre_ms=(t2 - t0) * 1000, A_det_ms=(t3 - t2) * 1000)
    # 전처리 후 (F) — 시간 표본은 캐시 없이 처음부터
    t0 = time.perf_counter()
    meta, im2, mask, g2 = R.prepare(path, pre_cfg, det, None if row["timed"] else SP / "b_cache")
    if im2 is None:
        row.update(F_n=0, F_area=0.0, F_box=0.0); out.append(row); continue
    im2, _ = R.apply_extra(im2, mask, rcF["extra"], set(group.get(path.name, "").split(";")))
    t1 = time.perf_counter()
    dF = [d for d in R.detect(im2, cfgF) if d["type"] == "crack" and R.road_overlap(mask, d["bbox"]) >= det["integration"]["min_road_overlap"]]
    t2 = time.perf_counter()
    s2 = g2["scale_x"] * g2["scale_y"]
    gF = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY); mk = (mask > 0).astype(np.uint8)
    row.update(F_edges=int(((cv2.Canny(gF, 100, 200) > 0) & (mk > 0)).sum()), F_kp=len(SIFT.detect(gF, mk)), F_px=int(mk.sum()))
    row.update(F_n=len(dF), F_area=sum(d["area"] for d in dF) / s2, F_box=sum(d["bbox"][2] * d["bbox"][3] for d in dF) / s2, F_pre_ms=(t1 - t0) * 1000, F_det_ms=(t2 - t1) * 1000)
    out.append(row)
    if i % 50 == 0: print(i, flush=True)
with open(SP / "diag/prepost.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in out for k in r))); w.writeheader(); w.writerows(out)
m = lambda k, rs: float(np.mean([r[k] for r in rs if k in r]))
T = [r for r in out if r["timed"]]
print("에지/장 A %.0f F %.0f · 1000px당 A %.2f F %.2f" % (m("A_edges", out), m("F_edges", out), 1000*sum(r["A_edges"] for r in out)/sum(r["A_px"] for r in out), 1000*sum(r.get("F_edges",0) for r in out)/max(1,sum(r.get("F_px",0) for r in out))))
print("특징점/장 A %.0f F %.0f · 1000px당 A %.2f F %.2f" % (m("A_kp", out), m("F_kp", out), 1000*sum(r["A_kp"] for r in out)/sum(r["A_px"] for r in out), 1000*sum(r.get("F_kp",0) for r in out)/max(1,sum(r.get("F_px",0) for r in out))))
print("후보/장 A %.1f F %.1f" % (m("A_n", out), m("F_n", out)))
print("후보 면적/장 (원본 px) A %.0f F %.0f · 후보 하나 A %.0f F %.0f" % (m("A_area", out), m("F_area", out),
      sum(r["A_area"] for r in out) / max(1, sum(r["A_n"] for r in out)), sum(r["F_area"] for r in out) / max(1, sum(r["F_n"] for r in out))))
print("후보 박스 면적/장 (원본 px) A %.0f F %.0f" % (m("A_box", out), m("F_box", out)))
print("시간/장 (표본 %d) A 전처리 %.0f + 검출 %.0f · F 전처리 %.0f + 검출 %.0f ms" % (len(T), m("A_pre_ms", T), m("A_det_ms", T), m("F_pre_ms", T), m("F_det_ms", T)))
