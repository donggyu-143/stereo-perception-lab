#!/usr/bin/env python3
"""
EXP-13 사전 확인: 무늬 면 한 쌍으로 조밀 시차(SGBM)가 잘 나오는지 빠르게 보기
==========================================================================
하는 일
1. 저장된 L/R 사진 한 쌍을 정렬(rectify)한다 (3단계 결과 사용)
2. StereoSGBM 으로 화면 전체의 시차 지도를 계산한다
3. ROI(네모) 안 시차의 중앙값으로 거리 Z = f·B/d 를 계산한다
4. [정렬된 왼쪽 영상 | 색칠한 시차 지도] 를 그림으로 저장한다
   - 색: 빨강 = 가까움, 파랑 = 멂, 검정 = 시차를 못 찾은 곳

평가 숫자 (ROI 안)
- 채움: 시차가 나온 픽셀 비율 (높을수록 좋음)
- 정답 근처: 중앙값 ±3px 안에 든 픽셀 비율 = "평면 위에서 맞는 값"으로 볼 수 있는 비율
- 사분위 폭: 시차 값이 얼마나 퍼졌는지 (평평한 판이면 0 에 가까워야 함)
- 좌/우 밝기: 두 카메라 노출 차이 확인용

--sweep: 여러 설정을 한 번에 돌려 표로 비교 (어떤 필터가 값을 지우는지 진단)
  SGBM 의 "버리는" 장치 3가지
  · uniquenessRatio : 1등 후보가 2등보다 N% 이상 좋아야 채택 (헷갈리면 버림)
  · disp12MaxDiff   : 왼→오 / 오→왼 결과가 N px 넘게 다르면 버림 (좌우 일관성 검사)
  · speckle         : 주변과 따로 노는 작은 덩어리를 버림
  --scale 0.5       : 영상을 절반으로 줄여서 계산 → 같은 창이 실제로 2배 넓은 면적을 봄,
                      웹캠 압축 잡음도 평균되어 줄어듦 (현업에서도 속도·안정성 위해 자주 씀)

사용 예
  python code/dense_preview.py --name EXP-13_dense_test --stereo EXP-10_stereo --roi 500 370 400 280
  python code/dense_preview.py --name EXP-13_dense_test --stereo EXP-10_stereo --roi 500 370 400 280 --sweep
"""
import argparse, glob, json, os
import cv2
import numpy as np

ROOT = os.path.expanduser("~/calibration_lab")


def run_sgbm(rL, rR, min_disp, num_disp, bs, uniq, lr, speckle, scale):
    """정렬된 회색 영상 한 쌍 → 원래 해상도 기준 시차 지도(px). 못 찾은 곳은 NaN."""
    if scale != 1.0:
        rL = cv2.resize(rL, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        rR = cv2.resize(rR, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    md = int(round(min_disp * scale))
    nd = int(np.ceil(num_disp * scale / 16) * 16)          # 16의 배수로 맞추기
    sgbm = cv2.StereoSGBM_create(minDisparity=md, numDisparities=nd, blockSize=bs,
                                 P1=8 * bs * bs, P2=32 * bs * bs, uniquenessRatio=uniq,
                                 speckleWindowSize=100 if speckle else 0, speckleRange=2,
                                 disp12MaxDiff=lr,                # -1 이면 좌우 검사 끔
                                 mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    d = sgbm.compute(rL, rR).astype(np.float32) / 16.0       # 16배 정수 → px
    d[d < md] = np.nan                                       # 못 찾은 곳
    if scale != 1.0:
        d = cv2.resize(d, (int(round(d.shape[1] / scale)), int(round(d.shape[0] / scale))),
                       interpolation=cv2.INTER_NEAREST) / scale   # 원래 해상도의 시차로 환산
    return d


def roi_stats(d, roi, fB):
    x, y, w, h = roi
    r = d[y:y + h, x:x + w]
    ok = r[np.isfinite(r)]
    fill = ok.size / r.size * 100
    if ok.size < 50:
        return fill, np.nan, np.nan, np.nan, np.nan
    med = np.median(ok)
    near = (np.abs(ok - med) <= 3).sum() / r.size * 100      # ROI 전체 대비 "정답 근처" 비율
    iqr = np.percentile(ok, 75) - np.percentile(ok, 25)
    return fill, near, med, iqr, fB / med


def save_fig(rL, d, roi, path):
    x, y, w, h = roi
    valid = np.isfinite(d)
    lo, hi = np.nanpercentile(d, 2), np.nanpercentile(d, 98)
    v = np.clip(np.nan_to_num((d - lo) / max(hi - lo, 1e-6) * 255), 0, 255).astype(np.uint8)
    col = cv2.applyColorMap(v, cv2.COLORMAP_JET)
    col[~valid] = 0
    left = cv2.cvtColor(rL, cv2.COLOR_GRAY2BGR)
    for im in (left, col):
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 255), 3)
    cv2.imwrite(path, np.hstack([left, col]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="captures/ 안의 폴더 (L, R)")
    ap.add_argument("--stereo", required=True, help="2단계 결과 json 이름 (예: EXP-10_stereo)")
    ap.add_argument("--pair", type=int, default=-1, help="몇 번째 쌍을 볼지 (기본: 마지막 쌍)")
    ap.add_argument("--roi", nargs=4, type=int, default=None, help="x y w h (정렬 영상 픽셀). 기본: 가운데 200x200")
    ap.add_argument("--min-disp", type=int, default=96, help="찾을 시차 최솟값 px (먼 쪽 한계)")
    ap.add_argument("--num-disp", type=int, default=352, help="시차 범위 폭 px, 16의 배수")
    ap.add_argument("--block", type=int, default=7, help="비교 창 크기 (홀수)")
    ap.add_argument("--uniq", type=int, default=10, help="uniquenessRatio (%)")
    ap.add_argument("--lr", type=int, default=1, help="disp12MaxDiff (px), -1 = 좌우 검사 끔")
    ap.add_argument("--no-speckle", action="store_true", help="작은 얼룩 제거 끔")
    ap.add_argument("--scale", type=float, default=1.0, help="계산 전 영상 축소 비율 (예: 0.5)")
    ap.add_argument("--sweep", action="store_true", help="여러 설정을 한 번에 비교")
    a = ap.parse_args()

    # ---- 캘리브레이션 결과 읽기 ----
    with open(os.path.join(ROOT, "results", f"{a.stereo}.json"), encoding="utf-8") as fp:
        s = json.load(fp)
    with open(os.path.join(ROOT, "results", f"{a.stereo}_rect.json"), encoding="utf-8") as fp:
        r = json.load(fp)
    KL, dL, KR, dR = (np.array(s[k]) for k in ("KL", "dL", "KR", "dR"))
    R1, R2, P1, P2 = (np.array(r[k]) for k in ("R1", "R2", "P1", "P2"))
    size = tuple(r["image_size"])
    f, B = P1[0, 0], r["baseline_mm"]
    fB = f * B

    # ---- 사진 한 쌍 읽기 + 정렬 ----
    filesL = sorted(glob.glob(os.path.join(ROOT, "captures", a.name, "L", "img_*.png")))
    if not filesL:
        raise SystemExit(f"[오류] captures/{a.name}/L 에 사진이 없습니다.")
    fL = filesL[a.pair]
    fR = os.path.join(ROOT, "captures", a.name, "R", os.path.basename(fL))
    gL = cv2.imread(fL, cv2.IMREAD_GRAYSCALE)
    gR = cv2.imread(fR, cv2.IMREAD_GRAYSCALE)
    mLx, mLy = cv2.initUndistortRectifyMap(KL, dL, R1, P1, size, cv2.CV_32FC1)
    mRx, mRy = cv2.initUndistortRectifyMap(KR, dR, R2, P2, size, cv2.CV_32FC1)
    rL = cv2.remap(gL, mLx, mLy, cv2.INTER_LINEAR)
    rR = cv2.remap(gR, mRx, mRy, cv2.INTER_LINEAR)

    H, W = rL.shape
    roi = tuple(a.roi) if a.roi else (W // 2 - 100, H // 2 - 100, 200, 200)
    x, y, w, h = roi
    print(f"사진: {os.path.basename(fL)}   f = {f:.1f}px, B = {B:.2f}mm, f·B = {fB:,.0f}")
    print(f"시차 탐색 범위: {a.min_disp} ~ {a.min_disp + a.num_disp}px "
          f"→ 거리 {fB/(a.min_disp + a.num_disp)/10:.0f} ~ {fB/a.min_disp/10:.0f}cm")
    # 같은 물체는 오른쪽 영상에서 시차만큼 왼쪽에 있으므로, 대략 같은 영역끼리 밝기 비교
    print(f"ROI 밝기: 왼쪽 {rL[y:y+h, x:x+w].mean():.0f}, 오른쪽 {rR[y:y+h, x:x+w].mean():.0f} (0~255)")

    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    if not a.sweep:
        d = run_sgbm(rL, rR, a.min_disp, a.num_disp, a.block, a.uniq, a.lr, not a.no_speckle, a.scale)
        fill, near, med, iqr, Z = roi_stats(d, roi, fB)
        print(f"ROI {roi}: 채움 {fill:.0f}%, 정답 근처 {near:.0f}%, 시차 중앙값 {med:.2f}px "
              f"(사분위 폭 {iqr:.2f}px) → 거리 Z = {Z/10:.1f}cm")
        out = os.path.join(ROOT, "figures", f"{a.name}_preview.png")
        save_fig(rL, d, roi, out)
        print(f"그림 저장: {out}")
        return

    # ---- 여러 설정 비교 ----
    #   (이름, 창, uniq, lr, speckle, scale)
    presets = [
        ("A 기본",                  7, 10,  1, True,  1.0),
        ("B 창15",                 15, 10,  1, True,  1.0),
        ("C 창15+uniq5",           15,  5,  1, True,  1.0),
        ("D 창15+uniq5+좌우끔",     15,  5, -1, True,  1.0),
        ("E 창15+필터 전부 끔",      15,  0, -1, False, 1.0),
        ("F 절반축소 창9",           9, 10,  1, True,  0.5),
        ("G 절반축소 창9+uniq5",     9,  5,  2, True,  0.5),
        ("H 1/4축소 창7+uniq5",      7,  5,  2, True,  0.25),
    ]
    print(f"\n{'설정':<22} {'채움':>6} {'정답근처':>8} {'중앙값':>8} {'사분위폭':>8} {'거리':>8}")
    for i, (nm, bs, uq, lr, sp, sc) in enumerate(presets):
        d = run_sgbm(rL, rR, a.min_disp, a.num_disp, bs, uq, lr, sp, sc)
        fill, near, med, iqr, Z = roi_stats(d, roi, fB)
        print(f"{nm:<22} {fill:5.0f}% {near:7.0f}% {med:7.1f}px {iqr:7.1f}px {Z/10:6.1f}cm")
        save_fig(rL, d, roi, os.path.join(ROOT, "figures", f"{a.name}_sweep_{nm.split()[0]}.png"))
    print(f"\n그림 저장: figures/{a.name}_sweep_A.png ~ _H.png")
    print("읽는 법: 필터를 끈 E 에서 채움이 크게 늘면 → 값은 계산되는데 '헷갈려서 버려지는' 것")
    print("        축소(F~H)에서 정답 근처가 늘면 → 무늬가 픽셀 대비 너무 듬성듬성하다는 뜻")


if __name__ == "__main__":
    main()
