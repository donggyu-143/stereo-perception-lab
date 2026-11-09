#!/usr/bin/env python3
"""
1단계 검증: 단안(카메라 한 대) 거리 검증 스크립트
================================================
질문: "캘리브레이션으로 구한 fx 가 실제와 맞는가?" → 줄자(정답)와 비교해서 확인한다

왜 필요한가
- 재투영 오차(RMS)는 fx 가 틀려도 잘 드러나지 않는다 (EXP-4 교차 검증에서 확인)
- fx 가 틀리면 계산한 거리가 같은 비율로 틀린다 → 거리를 줄자와 비교하면 fx 오류가 바로 드러난다

방법
1) 카메라를 책상에 고정하고, 줄자를 카메라 정면 방향으로 깔아 고정한다
2) 체커보드를 줄자의 여러 눈금(예: 50, 60, ... 120cm)에 차례로 세우고 한 장씩 찍는다
3) 각 사진에서 solvePnP 로 "카메라 기준 보드 중심의 3D 위치"를 구한다
4) 첫 위치에서 각 위치까지 보드가 움직인 거리(3D 직선 거리)를 계산해서
   줄자로 잰 이동 거리와 비교한다
   → 카메라 렌즈 중심이 케이스 안 어디인지 몰라도 된다 (차이를 보면 그 오차가 지워진다)
   → 줄자가 광축과 살짝 비스듬해도 된다 (3D 직선 거리를 쓰므로)
5) "계산 이동 거리 = s × 줄자 이동 거리" 로 직선을 맞춰 기울기 s 를 구한다
   - s = 1.000 이면 완벽
   - 거리는 fx 에 비례하므로 s ≈ (사용한 fx) / (진짜 fx)  →  진짜 fx ≈ 사용한 fx / s

사용 예
  # 촬영: 위치마다 SPACE 한 번 (가까운 위치부터 순서대로!)
  python code/calib_capture.py --cam 0 --name EXP-5_camA_distance
  # 계산: --tape 에 찍은 순서대로 줄자 눈금(cm)을 적는다
  python code/calib_distance.py --name EXP-5_camA_distance --tape 50 60 70 80 90 100 110 120 \
      --params EXP-2_camA_baseline_fixk3 EXP-3_camA_angles_fixk3
"""
import argparse, glob, json, os
import cv2
import numpy as np

ROOT = os.path.expanduser("~/calibration_lab")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="captures/ 안의 사진 폴더 (위치마다 1장, 찍은 순서대로)")
    ap.add_argument("--tape", nargs="+", type=float, required=True, help="각 사진의 줄자 눈금 (cm, 찍은 순서대로)")
    ap.add_argument("--params", nargs="+", required=True, help="비교할 결과 json 이름들 (results/ 안)")
    ap.add_argument("--square", type=float, default=25.0)
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--rows", type=int, default=6)
    a = ap.parse_args()
    pattern = (a.cols, a.rows)

    files = sorted(glob.glob(os.path.join(ROOT, "captures", a.name, "img_*.png")))
    if len(files) != len(a.tape):
        raise SystemExit(f"[오류] 사진은 {len(files)}장인데 줄자 눈금은 {len(a.tape)}개입니다. 개수를 맞춰 주세요.")
    tape_mm = np.array(a.tape) * 10.0                 # cm → mm

    # 보드 코너 3D 좌표 (mm) + 보드 중심점 (코너들의 가운데) → 이 점의 위치를 추적한다
    objp = np.zeros((a.cols * a.rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:a.cols, 0:a.rows].T.reshape(-1, 2) * a.square
    center = objp.mean(axis=0)                        # 보드 중심 (100, 62.5, 0) mm

    # 코너 검출 (calib_single.py 와 같은 방법)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = []
    for f in files:
        gray = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
        ok, c = cv2.findChessboardCorners(gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ok:
            raise SystemExit(f"[오류] 코너 검출 실패: {os.path.basename(f)} → 그 위치를 다시 찍어 주세요.")
        corners.append(cv2.cornerSubPix(gray, c, (11, 11), (-1, -1), crit))

    # 줄자 기준 "첫 위치에서 움직인 거리"
    d_tape = tape_mm - tape_mm[0]

    print(f"사진 {len(files)}장, 줄자 이동 거리: {', '.join(f'{v/10:.0f}' for v in d_tape)} cm")
    summary = {}
    for p in a.params:
        with open(os.path.join(ROOT, "results", f"{p}.json"), encoding="utf-8") as fp:
            r = json.load(fp)
        K, dist = np.array(r["K"]), np.array(r["dist"])

        # 각 사진에서 보드 자세를 구하고, 보드 중심의 카메라 기준 3D 위치를 계산
        P = []
        for c in corners:
            ok, rvec, tvec = cv2.solvePnP(objp, c, K, dist)
            R, _ = cv2.Rodrigues(rvec)
            P.append(R @ center + tvec.ravel())       # 보드 좌표 → 카메라 좌표: R·X + t
        P = np.array(P)

        # 첫 위치에서 각 위치까지의 3D 직선 거리 (계산값)
        d_est = np.linalg.norm(P - P[0], axis=1)

        # 원점을 지나는 직선 d_est = s · d_tape 의 기울기 s (최소제곱)
        #   s = Σ(d_tape·d_est) / Σ(d_tape²)
        s = float((d_tape * d_est).sum() / (d_tape ** 2).sum())
        err = d_est - d_tape                          # 위치별 오차 (mm)
        fx_used = K[0, 0]

        print(f"\n===== {p}  (fx = {fx_used:.1f}) =====")
        print(f"{'줄자 이동':>10} {'계산 이동':>10} {'오차':>9} {'오차율':>8}   카메라~보드 거리(참고)")
        for dt, de, e, z in zip(d_tape, d_est, err, P[:, 2]):
            rate = f"{e / dt * 100:+.2f}%" if dt > 0 else "-"
            print(f"{dt/10:>8.1f}cm {de/10:>8.1f}cm {e:>+7.1f}mm {rate:>8}   Z={z/10:.1f}cm")
        print(f"기울기 s = {s:.4f}  → 거리 배율 오차 {(s - 1) * 100:+.2f}%")
        print(f"이 결과로 추정한 '진짜' fx ≈ {fx_used / s:.1f}  (사용한 fx {fx_used:.1f})")
        summary[p] = {"fx_used": fx_used, "slope": s, "scale_error_pct": (s - 1) * 100,
                      "fx_implied": fx_used / s, "d_tape_mm": d_tape.tolist(),
                      "d_est_mm": d_est.tolist(), "err_mm": err.tolist()}

    out = os.path.join(ROOT, "results", f"{a.name}.json")
    with open(out, "w", encoding="utf-8") as fp:
        json.dump({"tape_cm": a.tape, "results": summary}, fp, ensure_ascii=False, indent=2)
    print(f"\n저장: results/{a.name}.json")
    print("※ 칸 크기(25.0mm)가 1% 틀리면 모든 결과가 똑같이 1% 틀린다 → 파라미터끼리의 '비교'에는 영향 없음")


if __name__ == "__main__":
    main()
