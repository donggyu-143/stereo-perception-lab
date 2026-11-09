#!/usr/bin/env python3
"""
2단계: 스테레오 캘리브레이션 계산 스크립트
==========================================
질문: "오른쪽 카메라는 왼쪽 카메라에 대해 어디에, 어떤 방향으로 놓여 있나?"
      → 회전 R (3x3) 과 이동 T (3x1, mm) = 두 카메라 사이의 외부 파라미터

방법
1) captures/<name>/L, R 의 같은 번호 사진 쌍에서 체커보드 코너를 찾는다
2) 1단계에서 구한 각 카메라의 내부 파라미터(K, dist)는 "고정"하고 (CALIB_FIX_INTRINSIC)
   R, T 만 계산한다 (stereoCalibrate)
   - 왜 고정? 내부 파라미터는 1단계에서 더 많은 사진과 줄자 검증으로 이미 확인했다.
     스테레오 사진은 두 카메라 모두에 보드가 보여야 해서 화면 가장자리 데이터가 부족하므로,
     여기서 내부 파라미터까지 다시 구하면 오히려 나빠질 수 있다
3) 결과
   - RMS: 두 카메라 사진 모두에 대한 재투영 오차
   - 베이스라인 |T| (mm) → 줄자로 잰 두 렌즈 간격과 비교 (검증)
   - 두 카메라의 각도 차이 (pitch/yaw/roll)
   - 쌍별 오차 → 튀는 쌍이 있으면 그 쌍만 빼고 다시 계산
   - results/<name>.json 저장 (3단계 정렬에서 읽어 씀)

사용 예
  python code/stereo_calib.py --name EXP-10_stereo \
      --params-left EXP-6_camA_refined_fixk3 --params-right EXP-7_camB_refined_fixk3
  # 특정 쌍 빼고 다시 계산:  --exclude 3 17
"""
import argparse, glob, json, os
from datetime import datetime
import cv2
import numpy as np

ROOT = os.path.expanduser("~/calibration_lab")


def load(name):
    with open(os.path.join(ROOT, "results", f"{name}.json"), encoding="utf-8") as fp:
        r = json.load(fp)
    return np.array(r["K"]), np.array(r["dist"]), tuple(r["image_size"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--params-left", required=True)
    ap.add_argument("--params-right", required=True)
    ap.add_argument("--exclude", nargs="*", type=int, default=[], help="제외할 쌍 번호들")
    ap.add_argument("--square", type=float, default=25.0)
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--tag", default="", help="결과 파일 이름 뒤에 붙일 말 (예: _excl)")
    a = ap.parse_args()
    pattern = (a.cols, a.rows)

    KL, dL, size = load(a.params_left)
    KR, dR, _ = load(a.params_right)

    objp = np.zeros((a.cols * a.rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:a.cols, 0:a.rows].T.reshape(-1, 2) * a.square
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def detect(path):
        g = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        ok, c = cv2.findChessboardCorners(g, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        return cv2.cornerSubPix(g, c, (11, 11), (-1, -1), crit) if ok else None

    filesL = sorted(glob.glob(os.path.join(ROOT, "captures", a.name, "L", "img_*.png")))
    obj, ptsL, ptsR, ids = [], [], [], []
    for fL in filesL:
        i = int(os.path.basename(fL)[4:7])                 # img_012.png → 12
        if i in a.exclude:
            continue
        fR = os.path.join(ROOT, "captures", a.name, "R", os.path.basename(fL))
        cL, cR = detect(fL), (detect(fR) if os.path.exists(fR) else None)
        if cL is None or cR is None:
            print(f"  코너 검출 실패, 제외: img_{i:03d}")
            continue
        obj.append(objp); ptsL.append(cL); ptsR.append(cR); ids.append(i)
    print(f"사용한 쌍: {len(ids)} / {len(filesL)}  (제외 지정: {a.exclude or '없음'})")
    if len(ids) < 5:
        raise SystemExit("[오류] 쌍이 너무 적습니다 (5쌍 이상 필요).")

    # ---- 핵심: 스테레오 캘리브레이션 ----
    #   CALIB_FIX_INTRINSIC: KL, dL, KR, dR 는 그대로 두고 R, T 만 구한다
    #   E (Essential), F (Fundamental): 두 카메라의 기하 관계를 나타내는 행렬 (3단계·검증에 사용)
    rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        obj, ptsL, ptsR, KL, dL, KR, dR, size, flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

    # 쌍별 오차: 각 쌍에서 왼쪽 보드 자세를 구하고, R,T 로 오른쪽 자세를 만들어 양쪽에 재투영
    per = []
    for o, cL, cR in zip(obj, ptsL, ptsR):
        _, rv, tv = cv2.solvePnP(o, cL, KL, dL)
        RL, _ = cv2.Rodrigues(rv)
        rvR, _ = cv2.Rodrigues(R @ RL)
        tvR = R @ tv + T
        pL, _ = cv2.projectPoints(o, rv, tv, KL, dL)
        pR, _ = cv2.projectPoints(o, rvR, tvR, KR, dR)
        e = np.concatenate([np.linalg.norm(pL.reshape(-1, 2) - cL.reshape(-1, 2), axis=1),
                            np.linalg.norm(pR.reshape(-1, 2) - cR.reshape(-1, 2), axis=1)])
        per.append(float(np.sqrt((e ** 2).mean())))
    per = np.array(per)

    ang = np.degrees(cv2.Rodrigues(R)[0].ravel())          # [pitch(x), yaw(y), roll(z)] 근사
    base = float(np.linalg.norm(T))
    print("\n===== 스테레오 캘리브레이션 결과 =====")
    print(f"재투영 오차 RMS : {rms:.4f} px  (쌍별 평균 {per.mean():.3f}, 최대 {per.max():.3f} = img_{ids[int(per.argmax())]:03d})")
    print(f"T (mm)          : [{T[0,0]:+.2f}, {T[1,0]:+.2f}, {T[2,0]:+.2f}]")
    print(f"베이스라인 |T|  : {base:.2f} mm   ← 줄자 실측과 비교")
    print(f"두 카메라 각도 차이: pitch {ang[0]:+.2f}°, yaw {ang[1]:+.2f}°, roll {ang[2]:+.2f}°")
    order = np.argsort(-per)[:5]
    print("오차 큰 쌍 Top5 : " + ", ".join(f"img_{ids[j]:03d}={per[j]:.3f}" for j in order))

    out = {"name": a.name, "time": datetime.now().isoformat(timespec="seconds"),
           "params_left": a.params_left, "params_right": a.params_right,
           "image_size": list(size), "n_pairs": len(ids), "excluded": a.exclude, "rms_px": rms,
           "R": R.tolist(), "T_mm": T.ravel().tolist(), "baseline_mm": base,
           "angles_deg_pitch_yaw_roll": ang.tolist(), "E": E.tolist(), "F": F.tolist(),
           "per_pair_rms_px": {f"img_{i:03d}": float(e) for i, e in zip(ids, per)},
           "KL": KL.tolist(), "dL": dL.tolist(), "KR": KR.tolist(), "dR": dR.tolist()}
    path = os.path.join(ROOT, "results", f"{a.name}{a.tag}.json")
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    print(f"\n저장: results/{a.name}{a.tag}.json")


if __name__ == "__main__":
    main()
