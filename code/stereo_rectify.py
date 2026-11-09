#!/usr/bin/env python3
"""
3단계: 스테레오 정렬(rectification) 계산과 확인 스크립트
=======================================================
질문: "두 영상을 같은 높이로 나란히 맞췄을 때, 같은 점이 정말 같은 가로줄에 오는가?"

정렬이란
- 두 카메라는 방향이 조금씩 다르고(yaw 1.1°, roll −0.9°), 주점·왜곡도 다르다
- 정렬은 두 영상을 "가상으로 회전·보정"해서, 마치 완벽히 나란한 두 카메라로 찍은 것처럼 만든다
- 그러면 왼쪽 영상의 한 점과 짝이 되는 오른쪽 점이 반드시 "같은 가로줄(같은 y)"에 있게 된다
  → 4단계에서 시차를 찾을 때 가로줄 하나만 뒤지면 된다 (빠르고 정확)

방법
1) 2단계 결과(EXP-10_stereo.json)의 K, dist, R, T 로 stereoRectify → 정렬 변환(R1, R2, P1, P2, Q) 계산
2) 정렬 지도(map)를 만들어 두 영상에 적용 (initUndistortRectifyMap + remap)
3) 검증: 2단계 사진 쌍에서 체커보드 코너를 찾고, 정렬 좌표로 옮겨서 같은 코너의 좌우 y 차이를 잰다
   → 0 에 가까울수록 정렬이 잘 된 것 (보통 0.5px 이하면 양호)
4) 결과 저장: results/<stereo>_rect.json (4단계에서 사용), figures/ 에 정렬 전후 그림

사용 예
  python code/stereo_rectify.py --stereo EXP-10_stereo --pairs EXP-10_stereo
"""
import argparse, glob, json, os
import cv2
import numpy as np

ROOT = os.path.expanduser("~/calibration_lab")
PATTERN = (9, 6)


def draw_lines(img, step=40, color=(0, 255, 255)):
    """가로선을 일정 간격으로 그린다 → 좌우 영상에서 같은 점이 같은 선 위에 있는지 눈으로 확인"""
    out = img.copy()
    for y in range(step, out.shape[0], step):
        cv2.line(out, (0, y), (out.shape[1], y), color, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stereo", required=True, help="2단계 결과 json 이름 (results/ 안, 예: EXP-10_stereo)")
    ap.add_argument("--pairs", required=True, help="검증에 쓸 사진 쌍 폴더 (captures/ 안, L·R 하위 폴더)")
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="0 = 검은 테두리 없이 유효 영역만 남김 / 1 = 원본 픽셀 전부 보이게(검은 테두리 생김)")
    a = ap.parse_args()

    with open(os.path.join(ROOT, "results", f"{a.stereo}.json"), encoding="utf-8") as fp:
        s = json.load(fp)
    KL, dL, KR, dR = (np.array(s[k]) for k in ("KL", "dL", "KR", "dR"))
    R, T = np.array(s["R"]), np.array(s["T_mm"]).reshape(3, 1)
    size = tuple(s["image_size"])                       # (1280, 720)

    # ---- 1) 정렬 변환 계산 ----
    # R1, R2: 각 카메라를 "가상으로" 돌리는 회전 (두 카메라가 같은 방향을 보게)
    # P1, P2: 정렬 후의 새 카메라 행렬 (두 카메라가 같은 f, 같은 cy 를 갖게 됨)
    # Q     : 시차 d → 3D 좌표로 바꾸는 행렬 (4·5단계에서 사용)
    # roi   : 정렬 후 유효한 영역 (검은 부분 제외)
    # CALIB_ZERO_DISPARITY: 정렬 후 두 영상의 주점을 같게 맞춤 (무한히 먼 점의 시차 = 0)
    R1, R2, P1, P2, Q, roiL, roiR = cv2.stereoRectify(
        KL, dL, KR, dR, size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=a.alpha)

    # ---- 2) 정렬 지도 만들기 (픽셀마다 "원본의 어디를 가져올지" 표) ----
    mapLx, mapLy = cv2.initUndistortRectifyMap(KL, dL, R1, P1, size, cv2.CV_32FC1)
    mapRx, mapRy = cv2.initUndistortRectifyMap(KR, dR, R2, P2, size, cv2.CV_32FC1)

    f_rect = P1[0, 0]
    base = -P2[0, 3] / P2[0, 0]                       # P2 의 [0,3] = −f·B → B 를 꺼냄 (mm)
    print(f"정렬 후 초점거리 f = {f_rect:.1f} px, 베이스라인 B = {base:.2f} mm, 주점 cy = {P1[1, 2]:.1f}")
    print(f"유효 영역 ROI  L = {roiL}, R = {roiR}  (x, y, 너비, 높이)")

    # ---- 3) 검증: 코너의 좌우 y 차이 ----
    filesL = sorted(glob.glob(os.path.join(ROOT, "captures", a.pairs, "L", "img_*.png")))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    dy_raw_all, dy_rect_all, disp_all, per_pair = [], [], [], []
    for fL in filesL:
        fR = os.path.join(ROOT, "captures", a.pairs, "R", os.path.basename(fL))
        gL = cv2.cvtColor(cv2.imread(fL), cv2.COLOR_BGR2GRAY)
        gR = cv2.cvtColor(cv2.imread(fR), cv2.COLOR_BGR2GRAY)
        okL, cL = cv2.findChessboardCorners(gL, PATTERN, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        okR, cR = cv2.findChessboardCorners(gR, PATTERN, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not (okL and okR):
            continue
        cL = cv2.cornerSubPix(gL, cL, (11, 11), (-1, -1), crit)
        cR = cv2.cornerSubPix(gR, cR, (11, 11), (-1, -1), crit)
        # 원본 코너 좌표 → 정렬 좌표: 왜곡 제거 + R1(R2) 회전 + P1(P2) 투영
        rL = cv2.undistortPoints(cL, KL, dL, R=R1, P=P1).reshape(-1, 2)
        rR = cv2.undistortPoints(cR, KR, dR, R=R2, P=P2).reshape(-1, 2)
        dy = rL[:, 1] - rR[:, 1]                          # 정렬 후 세로 차이 (0 이어야 함)
        dy_rect_all.append(dy)
        dy_raw_all.append(cL.reshape(-1, 2)[:, 1] - cR.reshape(-1, 2)[:, 1])   # 정렬 전 (참고)
        disp_all.append(rL[:, 0] - rR[:, 0])              # 시차 (4단계 미리보기)
        per_pair.append((os.path.basename(fL), float(np.sqrt((dy ** 2).mean()))))

    dy_rect = np.concatenate(dy_rect_all); dy_raw = np.concatenate(dy_raw_all); disp = np.concatenate(disp_all)
    print(f"\n검증에 쓴 쌍: {len(per_pair)}")
    print("===== 같은 코너의 좌우 세로 차이 (dy) =====")
    print(f"정렬 전 : 평균 {dy_raw.mean():+.1f} px, 절댓값 평균 {np.abs(dy_raw).mean():.1f} px  (주점 차이·방향 차이 때문)")
    print(f"정렬 후 : 평균 {dy_rect.mean():+.3f} px, RMS {np.sqrt((dy_rect**2).mean()):.3f} px, 최대 {np.abs(dy_rect).max():.3f} px")
    worst = sorted(per_pair, key=lambda x: -x[1])[:3]
    print("dy 큰 쌍 Top3 : " + ", ".join(f"{n}={e:.3f}" for n, e in worst))
    # 시차 → 거리 미리보기: Z = f·B/d
    print(f"시차 범위      : {disp.min():.0f} ~ {disp.max():.0f} px  → 거리 약 {f_rect*base/disp.max()/10:.0f} ~ {f_rect*base/disp.min()/10:.0f} cm")

    # ---- 4) 저장 ----
    out = {"stereo": a.stereo, "alpha": a.alpha, "image_size": list(size),
           "R1": R1.tolist(), "R2": R2.tolist(), "P1": P1.tolist(), "P2": P2.tolist(), "Q": Q.tolist(),
           "roiL": list(roiL), "roiR": list(roiR), "f_rect_px": float(f_rect), "baseline_mm": float(base),
           "dy_rect_rms_px": float(np.sqrt((dy_rect ** 2).mean())), "dy_rect_max_px": float(np.abs(dy_rect).max()),
           "dy_raw_mean_px": float(dy_raw.mean()), "per_pair_dy_rms_px": dict(per_pair)}
    with open(os.path.join(ROOT, "results", f"{a.stereo}_rect.json"), "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)

    # 그림: 첫 번째 쌍으로 정렬 전·후 비교 (가로선 = 같은 점이 같은 선 위에 있는지 확인)
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    name0 = per_pair[0][0]
    imL = cv2.imread(os.path.join(ROOT, "captures", a.pairs, "L", name0))
    imR = cv2.imread(os.path.join(ROOT, "captures", a.pairs, "R", name0))
    before = np.hstack([draw_lines(imL), draw_lines(imR)])
    reL = cv2.remap(imL, mapLx, mapLy, cv2.INTER_LINEAR)
    reR = cv2.remap(imR, mapRx, mapRy, cv2.INTER_LINEAR)
    after = np.hstack([draw_lines(reL, color=(0, 255, 0)), draw_lines(reR, color=(0, 255, 0))])
    for img, label in ((before, "BEFORE rectification"), (after, "AFTER rectification")):
        cv2.putText(img, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
    cv2.imwrite(os.path.join(ROOT, "figures", f"{a.stereo}_rect_before.png"), before)
    cv2.imwrite(os.path.join(ROOT, "figures", f"{a.stereo}_rect_after.png"), after)
    print(f"\n저장: results/{a.stereo}_rect.json")
    print(f"그림: figures/{a.stereo}_rect_before.png, _rect_after.png  ({name0})")


if __name__ == "__main__":
    main()
