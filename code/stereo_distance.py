#!/usr/bin/env python3
"""
4단계: 스테레오 거리 측정 + 줄자 검증 스크립트
=============================================
질문: "스테레오 카메라로 잰 거리가 줄자와 얼마나 맞는가?"

준비 (촬영)
- 스테레오 장치를 바닥에 두고, 줄자를 정면으로 깐다
- 왼쪽 카메라(L) 렌즈 앞면 바로 아래의 줄자 눈금을 읽어 둔다 → --lens-at (cm)
- 보드 판의 아래 앞모서리를 50, 60, ..., 150cm 눈금에 맞춰 세우고 위치마다 1쌍씩 촬영
  python code/stereo_capture.py --left 0 --right 6 --name EXP-12_stereo_distance

계산 (세 가지 방법으로 거리를 구해 비교)
  ① 스테레오-코너: 정렬된 좌우 영상에서 같은 코너의 시차로 삼각측량 (triangulatePoints) ← 주 결과
  ② 스테레오-조밀: 영상 전체의 시차 지도(StereoSGBM)에서 보드 영역의 시차 중앙값으로 거리 계산
                  → 6단계(YOLO)처럼 "코너 같은 표식이 없는 물체"를 잴 때 쓰는 방식
  ③ 단안(왼쪽 카메라만): 보드 크기를 알고 있으므로 solvePnP 로 거리 계산 (비교용)

평가
- 거리를 재는 기준점: 보드의 "맨 아래 코너 줄"의 가운데 (줄자에 맞춘 판 아래모서리에 가장 가까운 점)
- 절대 오차: 계산 거리 − (줄자 눈금 − 렌즈 위치)
- 직선 맞추기: 계산 거리 = s × 실제 거리 + b → s 가 1 에서 벗어난 만큼이 배율 오차 (f·B 의 오차),
  b 는 일정하게 더해지는 오차 (렌즈 위치 읽기 오차, 보드 기울기 등)

사용 예
  python code/stereo_distance.py --name EXP-12_stereo_distance --stereo EXP-10_stereo \
      --tape 50 60 70 80 90 100 110 120 130 140 150 --lens-at 0
"""
import argparse, glob, json, os
import cv2
import numpy as np

ROOT = os.path.expanduser("~/calibration_lab")
PATTERN, SQUARE = (9, 6), 25.0


def fit_line(x, y):
    """y = s·x + b 최소제곱 직선. 기울기 s, 절편 b, 잔차 RMS 를 돌려준다."""
    A = np.vstack([x, np.ones_like(x)]).T
    (s, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    return s, b, float(np.sqrt(((y - (s * x + b)) ** 2).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="captures/ 안의 거리 측정 사진 폴더 (L, R)")
    ap.add_argument("--stereo", required=True, help="2단계 결과 json (예: EXP-10_stereo). <stereo>_rect.json 도 사용")
    ap.add_argument("--tape", nargs="+", type=float, required=True, help="각 쌍의 줄자 눈금 cm (찍은 순서대로)")
    ap.add_argument("--lens-at", type=float, default=0.0, help="왼쪽 렌즈 앞면 바로 아래 줄자 눈금 (cm)")
    a = ap.parse_args()

    with open(os.path.join(ROOT, "results", f"{a.stereo}.json"), encoding="utf-8") as fp:
        s = json.load(fp)
    with open(os.path.join(ROOT, "results", f"{a.stereo}_rect.json"), encoding="utf-8") as fp:
        r = json.load(fp)
    KL, dL, KR, dR = (np.array(s[k]) for k in ("KL", "dL", "KR", "dR"))
    R1, R2, P1, P2 = (np.array(r[k]) for k in ("R1", "R2", "P1", "P2"))
    size = tuple(r["image_size"])
    f, B = P1[0, 0], r["baseline_mm"]

    # 정렬 지도 (조밀 시차 계산용)
    mLx, mLy = cv2.initUndistortRectifyMap(KL, dL, R1, P1, size, cv2.CV_32FC1)
    mRx, mRy = cv2.initUndistortRectifyMap(KR, dR, R2, P2, size, cv2.CV_32FC1)

    # 조밀 시차 계산기 (Semi-Global Block Matching)
    #   numDisparities: 찾을 시차 범위 (16의 배수). 가까운 50cm 에서 시차가 약 400px 라 448 로 넉넉히
    #   blockSize: 비교할 작은 창 크기 / P1, P2: 이웃 픽셀끼리 시차가 매끄럽게 이어지도록 하는 벌점
    #   uniquenessRatio: 두 번째로 좋은 후보보다 확실히 좋을 때만 채택 (체커보드처럼 반복 무늬에서 오답 줄이기)
    bs = 7
    sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=448, blockSize=bs,
                                 P1=8 * bs * bs, P2=32 * bs * bs, uniquenessRatio=10,
                                 speckleWindowSize=100, speckleRange=2, disp12MaxDiff=1,
                                 mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    filesL = sorted(glob.glob(os.path.join(ROOT, "captures", a.name, "L", "img_*.png")))
    if len(filesL) != len(a.tape):
        raise SystemExit(f"[오류] 사진 쌍 {len(filesL)}개, 줄자 눈금 {len(a.tape)}개 → 개수를 맞춰 주세요.")
    true_mm = (np.array(a.tape) - a.lens_at) * 10.0      # 렌즈 기준 실제 거리 (mm)

    Z_st, Z_dense, Z_mono, disp_corner = [], [], [], []
    fig_saved = False
    for i, fL in enumerate(filesL):
        fR = os.path.join(ROOT, "captures", a.name, "R", os.path.basename(fL))
        imL, imR = cv2.imread(fL), cv2.imread(fR)
        gL, gR = cv2.cvtColor(imL, cv2.COLOR_BGR2GRAY), cv2.cvtColor(imR, cv2.COLOR_BGR2GRAY)
        okL, cL = cv2.findChessboardCorners(gL, PATTERN, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        okR, cR = cv2.findChessboardCorners(gR, PATTERN, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not (okL and okR):
            raise SystemExit(f"[오류] 코너 검출 실패: {os.path.basename(fL)}")
        cL = cv2.cornerSubPix(gL, cL, (11, 11), (-1, -1), crit)
        cR = cv2.cornerSubPix(gR, cR, (11, 11), (-1, -1), crit)
        # 두 사진의 코너 순서가 반대로 잡히는 경우 맞춰 주기
        if np.linalg.norm(cL[0] - cR[0]) > np.linalg.norm(cL[0] - cR[-1]):
            cR = cR[::-1].copy()

        # "맨 아래 코너 줄" 고르기: 코너는 9개씩 6줄. 화면에서 y 가 가장 큰 줄
        rows = cL.reshape(PATTERN[1], PATTERN[0], 2)
        bottom = 0 if rows[0, :, 1].mean() > rows[-1, :, 1].mean() else PATTERN[1] - 1
        idx = np.arange(PATTERN[0]) + bottom * PATTERN[0]   # 그 줄 코너들의 번호

        # ---- ① 스테레오-코너 삼각측량 ----
        pL = cv2.undistortPoints(cL, KL, dL, R=R1, P=P1).reshape(-1, 2)   # 정렬 좌표
        pR = cv2.undistortPoints(cR, KR, dR, R=R2, P=P2).reshape(-1, 2)
        X = cv2.triangulatePoints(P1, P2, pL.T, pR.T)                     # 동차좌표 4xN
        X = (X[:3] / X[3]).T                                              # 3D (mm), 정렬된 왼쪽 카메라 기준
        Z_st.append(X[idx, 2].mean())
        disp_corner.append((pL[idx, 0] - pR[idx, 0]).mean())

        # ---- ② 스테레오-조밀 (SGBM) ----
        rL = cv2.remap(gL, mLx, mLy, cv2.INTER_LINEAR)
        rR = cv2.remap(gR, mRx, mRy, cv2.INTER_LINEAR)
        disp = sgbm.compute(rL, rR).astype(np.float32) / 16.0            # SGBM 결과는 16배 정수
        mask = np.zeros(gL.shape, np.uint8)                               # 보드 아래쪽 두 줄 영역
        band = np.vstack([pL[idx], pL[idx + (-PATTERN[0] if bottom else PATTERN[0])]])
        cv2.fillConvexPoly(mask, cv2.convexHull(band.astype(np.int32)), 255)
        vals = disp[(mask > 0) & (disp > 0)]
        Z_dense.append(f * B / np.median(vals) if len(vals) > 50 else np.nan)

        # ---- ③ 단안 (왼쪽 카메라 + 보드 크기) ----
        _, rv, tv = cv2.solvePnP(objp, cL, KL, dL)
        Rm, _ = cv2.Rodrigues(rv)
        Xm = (Rm @ objp[idx].T).T + tv.ravel()
        Z_mono.append(Xm[:, 2].mean())

        # 그림: 중간 위치의 시차 지도
        if not fig_saved and i == len(filesL) // 2:
            vis = disp.copy(); vis[vis <= 0] = np.nan
            v = np.nan_to_num((vis - np.nanpercentile(vis, 2)) / (np.nanpercentile(vis, 98) - np.nanpercentile(vis, 2)) * 255)
            col = cv2.applyColorMap(np.clip(v, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
            col[disp <= 0] = 0                                             # 시차를 못 찾은 곳 = 검정
            cv2.polylines(col, [cv2.convexHull(band.astype(np.int32))], True, (255, 255, 255), 2)
            os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
            cv2.imwrite(os.path.join(ROOT, "figures", f"{a.name}_disparity.png"),
                        np.hstack([cv2.cvtColor(rL, cv2.COLOR_GRAY2BGR), col]))
            fig_saved = True

    Z_st, Z_dense, Z_mono = map(np.array, (Z_st, Z_dense, Z_mono))
    print(f"정렬 후 f = {f:.1f}px, B = {B:.2f}mm, 렌즈 위치 = 줄자 {a.lens_at}cm")
    print(f"\n{'줄자':>6} {'실제':>7} | {'스테레오(코너)':>16} | {'스테레오(조밀)':>16} | {'단안(왼쪽)':>16} | 시차")
    for t, z0, z1, z2, z3, d in zip(a.tape, true_mm, Z_st, Z_dense, Z_mono, disp_corner):
        cell = lambda z: f"{z/10:6.1f}cm {(z-z0):+6.1f}mm" if np.isfinite(z) else f"{'실패':>16}"
        print(f"{t:5.0f}cm {z0/10:6.1f}cm | {cell(z1)} | {cell(z2)} | {cell(z3)} | {d:5.0f}px")

    summary = {}
    print("\n===== 방법별 요약 =====")
    for label, Z in (("stereo_corner", Z_st), ("stereo_dense", Z_dense), ("mono_left", Z_mono)):
        ok = np.isfinite(Z)
        err = Z[ok] - true_mm[ok]
        s_, b_, res = fit_line(true_mm[ok], Z[ok])
        rel = err / true_mm[ok] * 100
        print(f"{label:14s}: 절대 오차 평균 {err.mean():+6.1f}mm, |최대| {np.abs(err).max():5.1f}mm, "
              f"|오차율| 평균 {np.abs(rel).mean():.2f}% | 직선 s={s_:.4f} ({(s_-1)*100:+.2f}%), b={b_:+.1f}mm, 잔차 {res:.1f}mm")
        summary[label] = {"Z_mm": Z.tolist(), "err_mm": (Z - true_mm).tolist(), "slope": s_, "offset_mm": b_,
                          "residual_rms_mm": res, "mean_abs_err_pct": float(np.abs(rel).mean())}

    out = {"name": a.name, "stereo": a.stereo, "tape_cm": a.tape, "lens_at_cm": a.lens_at,
           "true_mm": true_mm.tolist(), "f_rect": f, "baseline_mm": B,
           "disparity_px": [float(d) for d in disp_corner], "methods": summary}
    with open(os.path.join(ROOT, "results", f"{a.name}.json"), "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    print(f"\n저장: results/{a.name}.json, 그림: figures/{a.name}_disparity.png")
    print("※ 절대 오차에는 '렌즈 위치 읽기'와 '보드 기울기' 오차가 일정하게 섞인다 → 직선의 b 로 드러남. 배율 오차는 s 로 본다")


if __name__ == "__main__":
    main()
