#!/usr/bin/env python3
"""
2단계: 스테레오 캘리브레이션용 동시 촬영 스크립트
================================================
하는 일
- 왼쪽(L)·오른쪽(R) 카메라를 동시에 열고, 체커보드 코너를 두 화면에서 실시간으로 찾는다
- "두 카메라 모두에서" 코너 54개가 검출됐을 때만 SPACE 로 저장된다
  → 저장 위치: captures/<name>/L/img_###.png, captures/<name>/R/img_###.png (같은 번호 = 같은 순간)
- 각 화면에 지금까지 찍은 보드 위치를 초록색으로 누적 표시한다 (촬영 범위 확인용)

1단계(calib_capture.py)와 다른 점
- 사진이 항상 L/R 한 쌍으로 저장된다. 한쪽이라도 보드를 못 찾으면 저장 안 됨
- 두 카메라가 같은 순간을 찍어야 하므로 grab → grab → retrieve → retrieve 순서로 받는다
  (웹캠은 하드웨어 동기화가 없으므로 촬영 순간 보드를 멈춰야 한다)
- 화면이 넓어서 표시만 줄여서 보여준다 (--view-scale). 저장은 원본 해상도

키: SPACE 저장 / u 마지막 쌍 취소 / q 종료

--any 옵션 (EXP-13 추가)
- 체커보드가 없어도 SPACE 로 저장된다 (랜덤 점 무늬처럼 코너가 없는 장면을 찍을 때)

사용 예
  python code/stereo_capture.py --left 0 --right 6 --name EXP-10_stereo
"""
import argparse, glob, os
import cv2
import numpy as np

PATTERN = (9, 6)


def open_cam(i, w, h, fps):
    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    # FPS 를 낮추면 두 카메라가 USB 대역폭을 덜 써서 "Corrupt JPEG data" (깨진 프레임) 가 줄어든다
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise SystemExit(f"[오류] /dev/video{i} 를 열 수 없습니다.")
    return cap


def find(gray):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    return cv2.findChessboardCorners(gray, PATTERN, flags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", type=int, required=True)
    ap.add_argument("--right", type=int, required=True)
    ap.add_argument("--name", required=True, help="실험 폴더 이름 (예: EXP-10_stereo)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=15, help="요청 FPS (낮을수록 깨진 프레임이 적음)")
    ap.add_argument("--view-scale", type=float, default=0.5)
    ap.add_argument("--any", action="store_true", help="코너 검출 없이도 저장 허용 (무늬 면 촬영용)")
    a = ap.parse_args()

    out = os.path.expanduser(f"~/calibration_lab/captures/{a.name}")
    os.makedirs(os.path.join(out, "L"), exist_ok=True)
    os.makedirs(os.path.join(out, "R"), exist_ok=True)
    capL = open_cam(a.left, a.width, a.height, a.fps)
    capR = open_cam(a.right, a.width, a.height, a.fps)

    covL = np.zeros((a.height, a.width, 3), np.uint8)   # 왼쪽 화면 누적 범위
    covR = np.zeros_like(covL)                          # 오른쪽 화면 누적 범위
    idx = len(glob.glob(os.path.join(out, "L", "img_*.png")))   # 이어서 찍으면 다음 번호부터
    history = []
    print(f"저장 폴더: {out}  (SPACE 저장 / u 취소 / q 종료)")

    while True:
        # 두 카메라를 최대한 같은 순간에 찍기: 먼저 둘 다 grab, 그다음 retrieve
        if not (capL.grab() and capR.grab()):
            continue
        okL, fL = capL.retrieve(); okR, fR = capR.retrieve()
        if not (okL and okR):
            continue
        gL, gR = cv2.cvtColor(fL, cv2.COLOR_BGR2GRAY), cv2.cvtColor(fR, cv2.COLOR_BGR2GRAY)
        foundL, cL = find(gL)
        foundR, cR = find(gR)
        both = foundL and foundR

        vL = cv2.addWeighted(fL, 1.0, covL, 0.35, 0)
        vR = cv2.addWeighted(fR, 1.0, covR, 0.35, 0)
        if foundL: cv2.drawChessboardCorners(vL, PATTERN, cL, True)
        if foundR: cv2.drawChessboardCorners(vR, PATTERN, cR, True)
        view = np.hstack([vL, vR])
        msg = "BOTH FOUND - SPACE to save" if both else "ANY MODE - SPACE to save" if a.any else f"L:{'ok' if foundL else '--'}  R:{'ok' if foundR else '--'}"
        cv2.putText(view, f"{msg} | saved {idx}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                    (0, 255, 0) if (both or a.any) else (0, 0, 255), 4)
        cv2.imshow("stereo_capture", cv2.resize(view, None, fx=a.view_scale, fy=a.view_scale, interpolation=cv2.INTER_AREA))

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord(" ") and (both or a.any):
            pL = os.path.join(out, "L", f"img_{idx:03d}.png")
            pR = os.path.join(out, "R", f"img_{idx:03d}.png")
            cv2.imwrite(pL, fL); cv2.imwrite(pR, fR)          # 원본 저장 (표시용 그림 없음)
            layers = []
            for cov, c in ((covL, cL), (covR, cR)):
                layer = np.zeros_like(cov)
                if both:   # 코너가 있을 때만 촬영 범위 표시 (--any 로 찍은 무늬 사진은 표시 없음)
                    cv2.fillConvexPoly(layer, cv2.convexHull(c.astype(np.int32)), (0, 180, 0))
                layers.append(layer)
            covL, covR = cv2.max(covL, layers[0]), cv2.max(covR, layers[1])
            history.append((pL, pR, layers))
            idx += 1
            print(f"저장 {idx}쌍: img_{idx-1:03d}.png")
        if k == ord("u") and history:
            pL, pR, _ = history.pop()
            os.remove(pL); os.remove(pR); idx -= 1
            covL, covR = np.zeros_like(covL), np.zeros_like(covR)
            for _, _, (lL, lR) in history:
                covL, covR = cv2.max(covL, lL), cv2.max(covR, lR)
            print(f"취소: {os.path.basename(pL)}")

    capL.release(); capR.release(); cv2.destroyAllWindows()
    print(f"총 {idx}쌍 저장: {out}")


if __name__ == "__main__":
    main()
