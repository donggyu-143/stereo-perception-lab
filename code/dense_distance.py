#!/usr/bin/env python3
"""
EXP-13 본 실험: 조밀 시차(SGBM) 거리 측정 vs 체커보드 코너 거리 (같은 자리, 같은 판)
=====================================================================================
질문: "코너 같은 표식이 없는 면도, 조밀 시차로 코너 방식만큼 정확하게 잴 수 있나?"

촬영 (한 위치에 2쌍씩, 순서 중요)
  위치마다:  ① 체커보드 면이 카메라를 보게 → SPACE   (짝수 번호: img_000, 002, ...)
             ② 판을 그 자리에서 뒤집어 무늬 면 → SPACE (홀수 번호: img_001, 003, ...)
  python code/stereo_capture.py --left 0 --right 6 --name EXP-13_dense_distance --any

정답 만들기
- ①의 체커보드 54개 코너를 삼각측량 → 보드 판 위 점들의 3D 위치(mm)
- 그 코너들이 차지하는 영역(정렬 영상 위 볼록 다각형)을 조금 안쪽으로 줄여 ROI 로 사용
- 정답 거리 = 코너들의 Z 평균 (보드 가운데 부근의 Z)
- ②에서 무늬 면은 거의 같은 평면(판 두께 2mm)이므로, 같은 ROI 의 조밀 시차 중앙값으로 Z 를 계산해 비교
  → 같은 영역을 비교하므로 카메라 기울기(pitch) 영향은 양쪽에 똑같이 들어가 상쇄된다

비교하는 조밀 설정
- 원본(창7) / 1/2 축소(창9) / 1/4 축소(창7)  ※ 사전 확인 결과 1/4 축소가 기본

출력
- 위치별 표: 줄자, 코너 Z, 각 설정의 채움·정답근처·Z·오차(mm, %)
- 각 설정의 직선 맞추기: 조밀 Z = s × 코너 Z + b
- figures/<name>_dense_grid.png : 가까운/중간/먼 위치의 시차 지도 (1/4 축소)
- results/<name>.json

사용 예
  python code/dense_distance.py --name EXP-13_dense_distance --stereo EXP-10_stereo \
      --tape 50 60 70 80 90 100 110 120 130 140 150
"""
import argparse, glob, json, os
import cv2
import numpy as np

ROOT = os.path.expanduser("~/calibration_lab")
PATTERN = (9, 6)
SETTINGS = [  # (이름, 축소, 창, uniq)
    ("원본", 1.0, 7, 5),
    ("1/2", 0.5, 9, 5),
    ("1/4", 0.25, 7, 5),
]


def run_sgbm(rL, rR, min_disp, num_disp, scale, bs, uniq):
    """정렬 영상 → 원래 해상도 기준 시차(px), 못 찾은 곳 NaN (dense_preview.py 와 같은 방식)"""
    if scale != 1.0:
        rL = cv2.resize(rL, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        rR = cv2.resize(rR, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    md = int(round(min_disp * scale))
    nd = int(np.ceil(num_disp * scale / 16) * 16)
    sgbm = cv2.StereoSGBM_create(minDisparity=md, numDisparities=nd, blockSize=bs,
                                 P1=8 * bs * bs, P2=32 * bs * bs, uniquenessRatio=uniq,
                                 speckleWindowSize=100, speckleRange=2, disp12MaxDiff=2,
                                 mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    d = sgbm.compute(rL, rR).astype(np.float32) / 16.0
    d[d < md] = np.nan
    if scale != 1.0:
        d = cv2.resize(d, (int(round(d.shape[1] / scale)), int(round(d.shape[0] / scale))),
                       interpolation=cv2.INTER_NEAREST) / scale
    return d


def colorize(d):
    lo, hi = np.nanpercentile(d, 2), np.nanpercentile(d, 98)
    v = np.clip(np.nan_to_num((d - lo) / max(hi - lo, 1e-6) * 255), 0, 255).astype(np.uint8)
    col = cv2.applyColorMap(v, cv2.COLORMAP_JET)
    col[~np.isfinite(d)] = 0
    return col


def fit_line(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    (s, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    return s, b, float(np.sqrt(((y - (s * x + b)) ** 2).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--stereo", required=True, help="예: EXP-10_stereo (<stereo>_rect.json 도 사용)")
    ap.add_argument("--tape", nargs="+", type=float, required=True, help="위치별 줄자 눈금 cm (기록용)")
    ap.add_argument("--min-disp", type=int, default=96)
    ap.add_argument("--num-disp", type=int, default=352)
    ap.add_argument("--shrink", type=float, default=0.8, help="ROI 를 코너 영역의 몇 배로 줄일지 (가장자리 제외)")
    a = ap.parse_args()

    with open(os.path.join(ROOT, "results", f"{a.stereo}.json"), encoding="utf-8") as fp:
        s = json.load(fp)
    with open(os.path.join(ROOT, "results", f"{a.stereo}_rect.json"), encoding="utf-8") as fp:
        r = json.load(fp)
    KL, dL, KR, dR = (np.array(s[k]) for k in ("KL", "dL", "KR", "dR"))
    R1, R2, P1, P2 = (np.array(r[k]) for k in ("R1", "R2", "P1", "P2"))
    size = tuple(r["image_size"])
    f, B = P1[0, 0], r["baseline_mm"]
    fB = f * B
    mLx, mLy = cv2.initUndistortRectifyMap(KL, dL, R1, P1, size, cv2.CV_32FC1)
    mRx, mRy = cv2.initUndistortRectifyMap(KR, dR, R2, P2, size, cv2.CV_32FC1)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    files = sorted(glob.glob(os.path.join(ROOT, "captures", a.name, "L", "img_*.png")))
    if len(files) != 2 * len(a.tape):
        raise SystemExit(f"[오류] 사진 쌍 {len(files)}개 ≠ 위치 {len(a.tape)}개 × 2. "
                         "위치마다 (체커보드 면, 무늬 면) 2쌍씩 찍었는지 확인하세요.")

    def load(fL):
        fR = os.path.join(ROOT, "captures", a.name, "R", os.path.basename(fL))
        gL = cv2.imread(fL, cv2.IMREAD_GRAYSCALE); gR = cv2.imread(fR, cv2.IMREAD_GRAYSCALE)
        return gL, gR

    rows, figs = [], []
    fig_pick = {0, len(a.tape) // 2, len(a.tape) - 1}          # 가까운/중간/먼 위치 그림
    for i, tape in enumerate(a.tape):
        # ---- ① 체커보드 면: 정답 Z 와 ROI ----
        gL, gR = load(files[2 * i])
        okL, cL = cv2.findChessboardCorners(gL, PATTERN, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        okR, cR = cv2.findChessboardCorners(gR, PATTERN, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not (okL and okR):
            raise SystemExit(f"[오류] {os.path.basename(files[2*i])} 에서 코너 검출 실패 "
                             "(체커보드 면이 짝수 번호인지, 너무 어둡지 않은지 확인)")
        cL = cv2.cornerSubPix(gL, cL, (11, 11), (-1, -1), crit)
        cR = cv2.cornerSubPix(gR, cR, (11, 11), (-1, -1), crit)
        if np.linalg.norm(cL[0] - cR[0]) > np.linalg.norm(cL[0] - cR[-1]):
            cR = cR[::-1].copy()
        pL = cv2.undistortPoints(cL, KL, dL, R=R1, P=P1).reshape(-1, 2)
        pR = cv2.undistortPoints(cR, KR, dR, R=R2, P=P2).reshape(-1, 2)
        X = cv2.triangulatePoints(P1, P2, pL.T, pR.T)
        X = (X[:3] / X[3]).T
        Z_true = X[:, 2].mean()                                    # 보드 가운데 부근 Z
        hull = cv2.convexHull(pL.astype(np.float32)).reshape(-1, 2)
        c = hull.mean(axis=0)
        hull = (c + (hull - c) * a.shrink).astype(np.int32)        # 가장자리 제외
        mask = np.zeros(gL.shape, np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        inside = mask > 0

        # ---- ② 무늬 면: 조밀 시차 ----
        gL, gR = load(files[2 * i + 1])
        rL = cv2.remap(gL, mLx, mLy, cv2.INTER_LINEAR)
        rR = cv2.remap(gR, mRx, mRy, cv2.INTER_LINEAR)
        row = {"tape_cm": tape, "Z_corner_mm": float(Z_true),
               "bright_L": float(rL[inside].mean()), "bright_R": float(rR[inside].mean())}
        for nm, sc, bs, uq in SETTINGS:
            d = run_sgbm(rL, rR, a.min_disp, a.num_disp, sc, bs, uq)
            v = d[inside]
            ok = v[np.isfinite(v)]
            fill = ok.size / v.size * 100
            if ok.size > 50:
                med = np.median(ok)
                Z = fB / med
                near = (np.abs(ok - med) <= 3).sum() / v.size * 100   # 중앙값 ±3px(원래 해상도) 비율
                iqr = float(np.percentile(ok, 75) - np.percentile(ok, 25))
            else:
                Z, near, iqr = np.nan, 0.0, np.nan
            row[nm] = {"fill": fill, "near": near, "Z_mm": float(Z), "iqr_px": iqr}
            if nm == "1/4" and i in fig_pick:
                col = colorize(d)
                left = cv2.cvtColor(rL, cv2.COLOR_GRAY2BGR)
                for im in (left, col):
                    cv2.polylines(im, [hull], True, (255, 255, 255), 3)
                    cv2.putText(im, f"{tape:.0f}cm", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)
                figs.append(np.hstack([left, col]))
        rows.append(row)

    # ---- 표 출력 ----
    print(f"f = {f:.1f}px, B = {B:.2f}mm / 정답 = 같은 자리 체커보드 코너 Z 평균")
    head = f"{'줄자':>5} {'코너Z':>8} {'밝기L/R':>8} |"
    for nm, *_ in SETTINGS:
        head += f" {nm+' 채움/근처':>11} {'오차':>8} |"
    print("\n" + head)
    for row in rows:
        line = f"{row['tape_cm']:4.0f}cm {row['Z_corner_mm']/10:6.1f}cm {row['bright_L']:3.0f}/{row['bright_R']:3.0f} |"
        for nm, *_ in SETTINGS:
            q = row[nm]
            if np.isfinite(q["Z_mm"]):
                err = q["Z_mm"] - row["Z_corner_mm"]
                line += f" {q['fill']:4.0f}%/{q['near']:3.0f}% {err:+6.1f}mm |"
            else:
                line += f" {q['fill']:4.0f}%/  0% {'실패':>8} |"
        print(line)

    print("\n설정별 요약 (조밀 Z = s × 코너 Z + b)")
    Zc = np.array([row["Z_corner_mm"] for row in rows])
    summary = {}
    for nm, *_ in SETTINGS:
        Zd = np.array([row[nm]["Z_mm"] for row in rows])
        ok = np.isfinite(Zd)
        if ok.sum() < 3:
            print(f"  {nm:>4}: 성공 위치 {ok.sum()}/{len(rows)} → 요약 불가")
            continue
        err = Zd[ok] - Zc[ok]
        pct = err / Zc[ok] * 100
        sl, b, res = fit_line(Zc[ok], Zd[ok])
        summary[nm] = {"n_ok": int(ok.sum()), "mean_err_mm": float(err.mean()), "max_abs_err_mm": float(np.abs(err).max()),
                       "max_abs_err_pct": float(np.abs(pct).max()), "s": float(sl), "b_mm": float(b), "resid_mm": res}
        print(f"  {nm:>4}: 성공 {ok.sum()}/{len(rows)}, 평균 오차 {err.mean():+.1f}mm, "
              f"최대 |오차| {np.abs(err).max():.1f}mm ({np.abs(pct).max():.2f}%), "
              f"s = {sl:.4f} ({(sl-1)*100:+.2f}%), b = {b:+.1f}mm, 잔차 {res:.1f}mm")

    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    if figs:
        out = os.path.join(ROOT, "figures", f"{a.name}_dense_grid.png")
        cv2.imwrite(out, cv2.resize(np.vstack(figs), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
        print(f"\n그림 저장: {out}")
    with open(os.path.join(ROOT, "results", f"{a.name}.json"), "w", encoding="utf-8") as fp:
        json.dump({"name": a.name, "stereo": a.stereo, "rows": rows, "summary": summary}, fp, ensure_ascii=False, indent=1)
    print(f"결과 저장: results/{a.name}.json")


if __name__ == "__main__":
    main()
