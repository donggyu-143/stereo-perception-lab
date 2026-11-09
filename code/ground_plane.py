#!/usr/bin/env python3
"""
5단계(EXP-14): 포인트 클라우드 + 바닥 평면 추정 → 카메라 높이·기울기 측정
=========================================================================
질문: "스테레오 깊이만으로 바닥(도로) 평면을 찾고, 카메라가 바닥에서 몇 mm 높이에서
       몇 도 기울어져 있는지 잴 수 있나? EXP-12에서 구한 기울기(약 5.5°)와 맞나?"

자율주행과의 연결
- 차량은 짐·제동·노면 때문에 카메라 기울기(pitch)가 계속 변한다 → 도로 평면을 보고 다시 추정
- 도로 평면을 알면: 단안 거리 계산, 주행 가능 영역(평면 위) / 장애물(평면 위로 솟은 것) 구분,
  위에서 본 지도(BEV) 변환, LiDAR 의 바닥 점 제거(ground removal) 등에 쓰인다

처리 순서
1. L/R 한 쌍 정렬 → SGBM(기본 1/4 축소, EXP-13 에서 정한 설정)으로 시차 지도
2. reprojectImageTo3D(시차, Q) → 픽셀마다 3D 점 (X, Y, Z mm) = 포인트 클라우드
   좌표계: 정렬된 왼쪽 카메라 기준 (X 오른쪽, Y 아래, Z 앞) ← EXP-12 와 같은 좌표계
3. 바닥 후보 영역(화면 아래쪽)의 점들로 RANSAC 평면 맞추기
   - RANSAC: 점 3개를 무작위로 골라 평면을 만들고, 그 평면에 가까운(±thr) 점이 가장 많은 평면을 고른다
     → 바닥 위 물건·잘못 잡힌 시차(이상치)가 섞여 있어도 바닥만 골라낼 수 있다
   - 고른 평면에 가까운 점들(inlier)로 최소제곱(SVD) 다시 맞춰 정밀화
4. 평면 n·X + d = 0 (n 은 바닥에서 카메라 쪽을 향하는 단위 법선)에서
   - 카메라 높이 = d (카메라 중심(원점)에서 평면까지 거리, mm)
   - pitch(위를 본 각도) = asin(n · 광축)   광축 = (0, 0, 1)
       수평이면 0°, 카메라가 위를 보면 + (광축이 바닥에서 멀어지는 방향)
   - roll(좌우 기울기) = atan2(n_x, −n_y)
5. 평탄도: inlier 의 평면까지 거리 RMS(mm), 거리 구간별 RMS
   → 바닥이 평평하다면 이 값이 곧 "조밀 깊이의 흔들림(정밀도)"
   → 거리가 멀수록 휘어 보이면(구간별 평균이 한쪽으로 쏠림) 캘리브레이션(f·B, 왜곡) 오차의 신호

선택(--checker-floor): 체커보드를 바닥에 눕혀 찍은 쌍이 있으면, 코너 삼각측량으로 평면을 따로 구해 비교
  (옵션 없이 실행하면 체커보드가 화면에 있어도 무시하고 조밀 방식으로만 계산)

출력
- 쌍마다: 점 개수, inlier 비율, 높이, pitch, roll, 평탄도
- 여러 쌍이면 평균 ± 표준편차 (반복성)
- figures/<name>_ground_img_000.png 등 (쌍마다): [바닥 inlier 영상 | 옆에서 본 그림 | 위에서 본 그림(바닥 점 분포)]
- results/<name>_cloud.ply  : 포인트 클라우드 파일 (MeshLab, CloudCompare 등으로 3D 보기)
- results/<name>.json

촬영 조건 (중요)
- 바닥 시차의 세로 변화율 ≈ B/h (px/행). 카메라가 낮으면(h ≈ 8cm, B 12cm → 1.5px/행) 비교 창 안에서
  시차가 크게 변해 매칭이 무너진다 → 장치를 높이 h ≥ 약 40cm 에 올리고 아래로 기울여 바닥을 보게 한다
  (차량 카메라: B ≈ 0.3m, h ≈ 1.3m → 0.23px/행. 높이 45cm 장치는 0.27px/행으로 비슷)
- 장치 전체를 움직이는 것은 괜찮다 (두 카메라 사이 관계 R, T 는 그대로)

사용 예
  python code/ground_plane.py --name EXP-14_ground --stereo EXP-10_stereo --min-disp 48 --zmax 3500 \
      --height-mm 450 --ref-pitch -9.5
"""
import argparse, glob, json, os
import cv2
import numpy as np

ROOT = os.path.expanduser("~/calibration_lab")
PATTERN = (9, 6)


def run_sgbm(rL, rR, min_disp, num_disp, scale, bs, uniq):
    """정렬 영상 → 원래 해상도 기준 시차(px). 못 찾은 곳은 NaN (dense_preview.py 와 같은 방식)"""
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


def fit_plane_svd(P):
    """점들(N×3)에 최소제곱 평면. 법선 n = 점 분포에서 가장 얇은 방향(SVD 마지막 축)."""
    c = P.mean(axis=0)
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    n = Vt[-1]
    return n, -n @ c


def ransac_plane(P, thr, iters, rng, max_tilt=40.0):
    """RANSAC: 점 3개로 만든 평면 중 ±thr 안에 드는 점이 가장 많은 것을 고르고, 그 점들로 다시 맞춤.
    max_tilt: 법선이 카메라의 '위'(−Y) 방향에서 이 각도 이상 벗어난 평면은 후보에서 뺀다
              → 가까이 세워둔 판·벽 같은 수직면을 바닥으로 잘못 고르는 것을 막는다 (바닥은 대략 아래쪽에 누워 있음)"""
    cos_lim = np.cos(np.radians(max_tilt))
    best = None
    N = len(P)
    for _ in range(iters):
        a, b, c = P[rng.choice(N, 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-6:            # 세 점이 거의 일직선이면 평면이 안 정해짐
            continue
        n /= norm
        if abs(n[1]) < cos_lim:    # 바닥이라기엔 너무 서 있는 평면
            continue
        d = -n @ a
        inl = np.abs(P @ n + d) < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None:
        raise RuntimeError("바닥 후보 평면을 찾지 못함 (--max-tilt, ROI 확인)")
    n, d = fit_plane_svd(P[best])                    # inlier 로 정밀화
    inl = np.abs(P @ n + d) < thr
    n, d = fit_plane_svd(P[inl])
    return n, d, inl


def orient(n, d):
    """법선이 바닥 → 카메라 쪽을 향하게(원점이 + 쪽, 즉 d > 0) 부호를 맞춘다."""
    return (n, d) if d > 0 else (-n, -d)


def pose_from_plane(n, d):
    pitch = np.degrees(np.arcsin(np.clip(n[2], -1, 1)))     # n · (0,0,1)
    roll = np.degrees(np.arctan2(n[0], -n[1]))
    return d, pitch, roll


def write_ply(path, P, C):
    with open(path, "w") as fp:
        fp.write("ply\nformat ascii 1.0\n")
        fp.write(f"element vertex {len(P)}\nproperty float x\nproperty float y\nproperty float z\n")
        fp.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for (x, y, z), (r, g, b) in zip(P, C):
            fp.write(f"{x:.1f} {y:.1f} {z:.1f} {r} {g} {b}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="captures/ 안의 폴더 (L, R)")
    ap.add_argument("--stereo", required=True, help="예: EXP-10_stereo (<stereo>_rect.json 도 사용)")
    ap.add_argument("--scale", type=float, default=0.25, help="SGBM 계산 축소 비율 (EXP-13 결론: 0.25)")
    ap.add_argument("--block", type=int, default=7)
    ap.add_argument("--min-disp", type=int, default=64, help="찾을 시차 최솟값 px (먼 쪽 한계, 64 → 약 3m)")
    ap.add_argument("--num-disp", type=int, default=400, help="시차 범위 폭 px")
    ap.add_argument("--roi", nargs=4, type=int, default=None,
                    help="바닥 후보 영역 x y w h (정렬 영상). 기본: x 470~끝, 화면 아래 40%")
    ap.add_argument("--zmax", type=float, default=2500, help="바닥 후보로 쓸 최대 거리 mm")
    ap.add_argument("--thr", type=float, default=8.0, help="RANSAC inlier 기준 (평면에서 ±mm)")
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--max-tilt", type=float, default=40.0, help="바닥 법선이 카메라 위쪽(−Y)에서 벗어날 수 있는 최대 각도")
    ap.add_argument("--height-mm", type=float, default=None, help="자로 잰 바닥~렌즈 중심 높이 (비교용)")
    ap.add_argument("--ref-pitch", type=float, default=None, help="비교할 기울기 (예: EXP-12 의 5.56)")
    ap.add_argument("--checker-floor", action="store_true",
                    help="체커보드를 바닥에 '눕혀' 찍은 쌍이 있을 때만 사용 (코너로 평면 계산). "
                         "세워둔 체커보드가 보이면 벽 같은 수직면이 잡히므로 기본은 꺼둠")
    a = ap.parse_args()

    with open(os.path.join(ROOT, "results", f"{a.stereo}.json"), encoding="utf-8") as fp:
        s = json.load(fp)
    with open(os.path.join(ROOT, "results", f"{a.stereo}_rect.json"), encoding="utf-8") as fp:
        r = json.load(fp)
    KL, dL, KR, dR = (np.array(s[k]) for k in ("KL", "dL", "KR", "dR"))
    R1, R2, P1, P2, Q = (np.array(r[k]) for k in ("R1", "R2", "P1", "P2", "Q"))
    size = tuple(r["image_size"])
    mLx, mLy = cv2.initUndistortRectifyMap(KL, dL, R1, P1, size, cv2.CV_32FC1)
    mRx, mRy = cv2.initUndistortRectifyMap(KR, dR, R2, P2, size, cv2.CV_32FC1)
    W, H = size
    x0, y0, w, h = a.roi if a.roi else (a.min_disp + a.num_disp + 10, int(H * 0.6), W, H)
    rng = np.random.default_rng(0)

    files = sorted(glob.glob(os.path.join(ROOT, "captures", a.name, "L", "img_*.png")))
    if not files:
        raise SystemExit(f"[오류] captures/{a.name}/L 에 사진이 없습니다.")

    rows, figs = [], []
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for fL in files:
        fR = os.path.join(ROOT, "captures", a.name, "R", os.path.basename(fL))
        gL = cv2.imread(fL, cv2.IMREAD_GRAYSCALE); gR = cv2.imread(fR, cv2.IMREAD_GRAYSCALE)
        rL = cv2.remap(gL, mLx, mLy, cv2.INTER_LINEAR)
        rR = cv2.remap(gR, mRx, mRy, cv2.INTER_LINEAR)
        nm = os.path.basename(fL)

        # ---- (선택, --checker-floor) 바닥에 눕힌 체커보드가 보이면 코너로 평면 ----
        okL = okR = False
        if a.checker_floor:
            okL, cL = cv2.findChessboardCorners(gL, PATTERN, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
            okR, cR = cv2.findChessboardCorners(gR, PATTERN, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if okL and okR:
            cL = cv2.cornerSubPix(gL, cL, (11, 11), (-1, -1), crit)
            cR = cv2.cornerSubPix(gR, cR, (11, 11), (-1, -1), crit)
            if np.linalg.norm(cL[0] - cR[0]) > np.linalg.norm(cL[0] - cR[-1]):
                cR = cR[::-1].copy()
            pL = cv2.undistortPoints(cL, KL, dL, R=R1, P=P1).reshape(-1, 2)
            pR = cv2.undistortPoints(cR, KR, dR, R=R2, P=P2).reshape(-1, 2)
            X = cv2.triangulatePoints(P1, P2, pL.T, pR.T)
            X = (X[:3] / X[3]).T
            n, d = orient(*fit_plane_svd(X))
            hh, pt, rl = pose_from_plane(n, d)
            res = np.sqrt(((X @ n + d) ** 2).mean())
            rows.append({"file": nm, "method": "checker", "height_mm": hh, "pitch_deg": pt, "roll_deg": rl,
                         "flat_rms_mm": res, "n_points": len(X)})
            continue

        # ---- 조밀 시차 → 포인트 클라우드 ----
        disp = run_sgbm(rL, rR, a.min_disp, a.num_disp, a.scale, a.block, 5)
        pts = cv2.reprojectImageTo3D(np.nan_to_num(disp, nan=-1).astype(np.float32), Q)   # H×W×3 (mm)
        valid = np.isfinite(disp) & (pts[..., 2] > 0) & (pts[..., 2] < a.zmax)
        roi = np.zeros_like(valid); roi[y0:y0 + h, x0:x0 + w] = True
        cand = valid & roi
        P = pts[cand]
        if len(P) < 500:
            print(f"{nm}: 바닥 후보 점이 {len(P)}개뿐 → 건너뜀 (바닥 무늬·ROI 확인)")
            continue
        sub = P[rng.choice(len(P), min(len(P), 20000), replace=False)]   # RANSAC 은 일부 점으로 (속도)
        n, d, _ = ransac_plane(sub, a.thr, a.iters, rng, a.max_tilt)
        n, d = orient(n, d)
        dist = P @ n + d
        inl = np.abs(dist) < a.thr
        n, d = orient(*fit_plane_svd(P[inl]))
        dist = P @ n + d
        inl = np.abs(dist) < a.thr
        hh, pt, rl = pose_from_plane(n, d)

        # 앞쪽 거리 = 광축을 바닥에 투영한 방향의 좌표
        fdir = np.array([0, 0, 1.0]) - n[2] * n
        fdir /= np.linalg.norm(fdir)
        fwd = P @ fdir
        bins = {}
        for lo in range(500, 2500, 250):
            m = inl & (fwd >= lo) & (fwd < lo + 250)
            if m.sum() > 200:
                bins[f"{lo/10:.0f}-{(lo+250)/10:.0f}cm"] = (float(dist[m].mean()), float(np.sqrt((dist[m] ** 2).mean())), int(m.sum()))
        rows.append({"file": nm, "method": "dense", "height_mm": hh, "pitch_deg": pt, "roll_deg": rl,
                     "flat_rms_mm": float(np.sqrt((dist[inl] ** 2).mean())), "n_points": int(len(P)),
                     "inlier_pct": float(inl.mean() * 100), "bins": bins})

        figs.append((rL, cand, inl, fwd, dist, pts, valid, n, d, nm, P))   # 쌍마다 그림

    if not rows:
        raise SystemExit("[오류] 계산된 쌍이 없습니다.")

    # ---- 출력 ----
    print(f"좌표계: 정렬된 왼쪽 카메라 (EXP-12 와 같음) / SGBM 축소 {a.scale}, RANSAC ±{a.thr}mm")
    print(f"\n{'사진':<12}{'방법':>8}{'높이':>10}{'pitch':>9}{'roll':>8}{'평탄도RMS':>11}{'inlier':>9}")
    for q in rows:
        inl_s = f"{q['inlier_pct']:6.1f}%" if "inlier_pct" in q else "   -   "
        print(f"{q['file']:<12}{q['method']:>8}{q['height_mm']:8.1f}mm{q['pitch_deg']:+8.2f}°{q['roll_deg']:+7.2f}°"
              f"{q['flat_rms_mm']:9.2f}mm{inl_s:>9}")
    dense = [q for q in rows if q["method"] == "dense"]
    if dense:
        print("\n거리 구간별 (첫 조밀 쌍): 평면까지 평균 / RMS / 점 개수")
        for k, (m, rms, cnt) in dense[0]["bins"].items():
            print(f"  {k:>10}: {m:+6.2f}mm / {rms:5.2f}mm / {cnt}")
    summary = {}
    for meth in ("dense", "checker"):
        g = [q for q in rows if q["method"] == meth]
        if not g:
            continue
        hs = np.array([q["height_mm"] for q in g]); ps = np.array([q["pitch_deg"] for q in g])
        rs = np.array([q["roll_deg"] for q in g])
        summary[meth] = {"n": len(g), "height_mm": [hs.mean(), hs.std()], "pitch_deg": [ps.mean(), ps.std()],
                         "roll_deg": [rs.mean(), rs.std()]}
        print(f"\n[{meth}] {len(g)}쌍 평균: 높이 {hs.mean():.1f} ± {hs.std():.1f}mm, "
              f"pitch {ps.mean():+.2f} ± {ps.std():.2f}°, roll {rs.mean():+.2f} ± {rs.std():.2f}°")
        if a.height_mm:
            print(f"   자로 잰 높이 {a.height_mm:.0f}mm 대비: {hs.mean() - a.height_mm:+.1f}mm")
        if a.ref_pitch is not None:
            print(f"   비교 기울기 {a.ref_pitch:.2f}° 대비: {ps.mean() - a.ref_pitch:+.2f}°")

    if figs:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
        print()
        for rL, cand, inl, fwd, dist, pts, valid, n, d, nm, P in figs:
            vis = cv2.cvtColor(rL, cv2.COLOR_GRAY2RGB)
            mask = np.zeros(rL.shape, bool); idx = np.argwhere(cand)
            mask[idx[inl, 0], idx[inl, 1]] = True
            vis[mask] = (0.5 * vis[mask] + 0.5 * np.array([0, 200, 0])).astype(np.uint8)
            out_mask = np.zeros(rL.shape, bool); out_mask[idx[~inl, 0], idx[~inl, 1]] = True
            vis[out_mask] = (0.5 * vis[out_mask] + 0.5 * np.array([220, 0, 0])).astype(np.uint8)
            cv2.rectangle(vis, (x0, y0), (min(x0 + w, W) - 1, min(y0 + h, H) - 1), (255, 255, 0), 2)

            fig, ax = plt.subplots(1, 3, figsize=(20, 5), gridspec_kw={"width_ratios": [1.3, 1, 0.8]})
            ax[0].imshow(vis); ax[0].set_title(f"{nm}: green = floor inlier, red = outlier, yellow = ROI"); ax[0].axis("off")
            # ② 옆에서 본 그림: 화면 전체 유효 점(바닥 기준 높이 vs 앞 거리)
            fdir = np.array([0, 0, 1.0]) - n[2] * n; fdir /= np.linalg.norm(fdir)
            rdir = np.cross(fdir, n)                       # 바닥 위 좌우 방향
            A = pts[valid]
            A = A[rng.choice(len(A), min(len(A), 30000), replace=False)]
            ax[1].scatter(A @ fdir / 10, (A @ n + d) / 10, s=0.3, c="0.6")
            ax[1].scatter(fwd[inl][::5] / 10, dist[inl][::5] / 10, s=0.3, c="g")
            ax[1].plot([0], [d / 10], "r^", ms=9)
            ax[1].axhline(0, c="k", lw=0.8)
            ax[1].set_xlabel("forward distance along floor (cm)"); ax[1].set_ylabel("height above fitted floor (cm)")
            ax[1].set_xlim(0, a.zmax / 10); ax[1].set_ylim(-10, max(60, d / 10 + 10))
            ax[1].legend(handles=[Line2D([], [], ls="", marker="o", c="0.6", label="all points"),
                                  Line2D([], [], ls="", marker="o", c="g", label="floor inliers"),
                                  Line2D([], [], ls="", marker="^", c="r", label=f"camera ({d:.0f} mm)")],
                         loc="upper right")
            ax[1].set_title("side view")
            # ③ 위에서 본 그림: 바닥 inlier 가 좌우·앞뒤로 얼마나 넓게 퍼졌는지 (좁으면 기울기가 불안정)
            Pi = P[inl][::5]
            ax[2].scatter(Pi @ rdir / 10, Pi @ fdir / 10, s=0.3, c="g")
            ax[2].plot([0], [0], "r^", ms=9)
            ax[2].set_xlabel("left / right (cm)"); ax[2].set_ylabel("forward (cm)")
            ax[2].set_xlim(-150, 150); ax[2].set_ylim(0, a.zmax / 10); ax[2].set_aspect("equal")
            ax[2].set_title("top view (floor inliers)")
            plt.tight_layout()
            out = os.path.join(ROOT, "figures", f"{a.name}_ground_{os.path.splitext(nm)[0]}.png")
            plt.savefig(out, dpi=100); plt.close()
            print(f"그림 저장: {out}")
        rL, cand, inl, fwd, dist, pts, valid, n, d, nm, P = figs[0]

        # 포인트 클라우드 PLY (화면 전체 유효 점, 밝기로 색)
        g = cv2.remap(rL, np.arange(W, dtype=np.float32)[None].repeat(H, 0),
                      np.arange(H, dtype=np.float32)[:, None].repeat(W, 1), cv2.INTER_NEAREST)
        P = pts[valid]; Cg = g[valid]
        step = max(1, len(P) // 200000)
        ply = os.path.join(ROOT, "results", f"{a.name}_cloud.ply")
        write_ply(ply, P[::step], np.stack([Cg[::step]] * 3, axis=1))
        print(f"포인트 클라우드: {ply} ({len(P[::step])}점)")

    with open(os.path.join(ROOT, "results", f"{a.name}.json"), "w", encoding="utf-8") as fp:
        json.dump({"name": a.name, "rows": rows, "summary": summary,
                   "height_mm_ruler": a.height_mm, "ref_pitch": a.ref_pitch}, fp, ensure_ascii=False, indent=1,
                  default=float)
    print(f"결과 저장: results/{a.name}.json")


if __name__ == "__main__":
    main()
