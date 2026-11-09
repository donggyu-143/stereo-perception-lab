#!/usr/bin/env python3
"""
2단계 준비: 두 카메라 방향 맞추기 (실시간 정렬 확인) 스크립트
============================================================
하는 일
- 왼쪽(L)·오른쪽(R) 카메라를 동시에 열고, 체커보드가 두 화면에 모두 보이면
  두 카메라가 "서로 얼마나 다른 방향을 보고 있는지"를 각도(°)로 실시간 표시한다
  · pitch 차이 (위아래로 숙인 정도 차이)  ← 가장 중요
  · roll 차이  (옆으로 기울어진 정도 차이)
  · yaw 차이   (좌우로 돌아간 정도 차이)
  · 베이스라인 (두 카메라 사이 거리, mm) — 줄자 실측과 비교용
- 1단계에서 구한 각 카메라의 내부 파라미터(fx, cx, 왜곡)를 써서 계산한다
  → 카메라 B 의 주점(cy)이 90px 어긋나 있어도 화면 위치만 보고 착각하지 않게 보정됨

왜 맞추나
- 스테레오 정렬(rectification)이 어느 정도 틀어짐은 계산으로 바로잡아 주지만,
  틀어짐이 크면 두 화면이 함께 쓸 수 있는 영역이 줄고 보정 오차도 커진다
- 목표: pitch·roll 차이 1° 이내 (yaw 는 몇 도 정도 괜찮음)

방법
- 체커보드를 장치 정면 약 1m 에 세워 두고, 화면 숫자를 보면서 카메라 아래에 얇은 받침(종이·골판지)을 넣어 조절
- 숫자가 목표 안에 들어오면 q 로 종료. 이후 카메라는 절대 만지지 않는다

사용 예
  python code/stereo_align_check.py --left 0 --right 6 \
      --params-left EXP-6_camA_refined_fixk3 --params-right EXP-7_camB_refined_fixk3
"""
import argparse, json, os
import cv2
import numpy as np

ROOT = os.path.expanduser("~/calibration_lab")
PATTERN, SQUARE = (9, 6), 25.0


def load(name):
    """results/<name>.json 에서 K(카메라 행렬), dist(왜곡 계수)를 읽는다."""
    with open(os.path.join(ROOT, "results", f"{name}.json"), encoding="utf-8") as fp:
        r = json.load(fp)
    return np.array(r["K"]), np.array(r["dist"])


def open_cam(i, w, h):
    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if not cap.isOpened():
        raise SystemExit(f"[오류] /dev/video{i} 를 열 수 없습니다.")
    return cap


def board_pose(gray, K, dist, objp):
    """체커보드를 찾아 카메라 기준 보드의 회전 행렬 R, 위치 t 를 돌려준다 (못 찾으면 None)."""
    ok, c = cv2.findChessboardCorners(gray, PATTERN, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK)
    if not ok:
        return None, None
    c = cv2.cornerSubPix(gray, c, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
    _, rvec, tvec = cv2.solvePnP(objp, c, K, dist)     # 내부 파라미터는 고정, 보드 자세만 계산
    R, _ = cv2.Rodrigues(rvec)
    return (R, tvec.ravel()), c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", type=int, required=True)
    ap.add_argument("--right", type=int, required=True)
    ap.add_argument("--params-left", required=True, help="왼쪽 카메라 결과 json 이름 (results/ 안)")
    ap.add_argument("--params-right", required=True, help="오른쪽 카메라 결과 json 이름")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--view-scale", type=float, default=0.5)
    a = ap.parse_args()

    KL, dL = load(a.params_left); KR, dR = load(a.params_right)
    objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE
    capL, capR = open_cam(a.left, a.width, a.height), open_cam(a.right, a.width, a.height)
    hist = []   # 최근 값 평균 (숫자가 떨리지 않게)

    while True:
        capL.grab(); capR.grab()
        _, fL = capL.retrieve(); _, fR = capR.retrieve()
        gL, gR = cv2.cvtColor(fL, cv2.COLOR_BGR2GRAY), cv2.cvtColor(fR, cv2.COLOR_BGR2GRAY)
        pL, cL = board_pose(gL, KL, dL, objp)
        pR, cR = board_pose(gR, KR, dR, objp)
        if cL is not None: cv2.drawChessboardCorners(fL, PATTERN, cL, True)
        if cR is not None: cv2.drawChessboardCorners(fR, PATTERN, cR, True)
        view = np.hstack([fL, fR])

        if pL and pR:
            (RL, tL), (RR, tR) = pL, pR
            # 같은 보드를 두 카메라가 본 자세로부터, L 카메라에서 R 카메라로의 회전·이동을 구한다
            #   보드 → L: X_L = RL·X + tL,  보드 → R: X_R = RR·X + tR
            #   → X_R = (RR·RLᵀ)·X_L + (tR − RR·RLᵀ·tL)
            Rrel = RR @ RL.T
            t = tR - Rrel @ tL
            ang = np.degrees(cv2.Rodrigues(Rrel)[0].ravel())   # [x축=pitch, y축=yaw, z축=roll] 근사
            hist.append((*ang, np.linalg.norm(t)))
            hist = hist[-15:]
            pitch, yaw, roll, base = np.mean(hist, axis=0)
            ok = abs(pitch) < 1.0 and abs(roll) < 1.0
            col = (0, 200, 0) if ok else (0, 0, 255)
            lines = [f"pitch diff {pitch:+.2f} deg  (goal |x|<1)",
                     f"roll  diff {roll:+.2f} deg  (goal |x|<1)",
                     f"yaw   diff {yaw:+.2f} deg",
                     f"baseline   {base:.1f} mm",
                     "OK - stop adjusting" if ok else "adjust camera tilt"]
            for i, s in enumerate(lines):
                cv2.putText(view, s, (20, 50 + 45 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
        else:
            cv2.putText(view, "board must be visible in BOTH cameras", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            hist = []

        cv2.imshow("stereo_align_check (q: quit)", cv2.resize(view, None, fx=a.view_scale, fy=a.view_scale, interpolation=cv2.INTER_AREA))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if hist:
        pitch, yaw, roll, base = np.mean(hist, axis=0)
        print(f"마지막 값: pitch {pitch:+.2f}°, roll {roll:+.2f}°, yaw {yaw:+.2f}°, baseline {base:.1f} mm")
    capL.release(); capR.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
