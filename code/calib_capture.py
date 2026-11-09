#!/usr/bin/env python3
"""
1단계: 단일 캘리브레이션용 체커보드 촬영
- 카메라 1대를 열고 실시간으로 체커보드 코너를 검출해 화면에 표시한다
- 코너가 모두 검출된 상태에서 SPACE를 누를 때만 원본 사진을 저장한다
- 지금까지 저장한 사진의 체커보드 위치를 화면에 누적 표시한다 (가장자리까지 골고루 찍었는지 확인용)

키: SPACE 저장 / u 마지막 사진 취소 / q 종료

사용 예:
  python scripts/calib_capture.py --cam 4 --name EXP-2_camA_baseline
"""
import argparse, os, glob
import cv2
import numpy as np

PATTERN = (9, 6)  # 체커보드 내부 코너 (가로, 세로)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, required=True, help="/dev/videoN 의 N")
    ap.add_argument("--name", required=True, help="실험 폴더 이름 (예: EXP-2_camA_baseline)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    a = ap.parse_args()

    out = os.path.expanduser(f"~/calibration_lab/captures/{a.name}")
    os.makedirs(out, exist_ok=True)

    cap = cv2.VideoCapture(a.cam, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    if not cap.isOpened():
        raise SystemExit(f"[오류] /dev/video{a.cam} 를 열 수 없습니다.")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"/dev/video{a.cam}: {w}x{h}, 저장 폴더: {out}")

    coverage = np.zeros((h, w, 3), np.uint8)   # 저장한 보드 위치 누적
    saved = sorted(glob.glob(os.path.join(out, "img_*.png")))
    idx = len(saved)
    history = []
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, PATTERN, flags)

        view = cv2.addWeighted(frame, 1.0, coverage, 0.35, 0)
        if found:
            cv2.drawChessboardCorners(view, PATTERN, corners, found)
        status = "FOUND - SPACE to save" if found else "board not found"
        color = (0, 255, 0) if found else (0, 0, 255)
        cv2.putText(view, f"{status} | saved {idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(view, "green area = already captured. cover edges & corners, tilt the board",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        cv2.imshow(f"calib_capture /dev/video{a.cam}", view)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord(" ") and found:
            path = os.path.join(out, f"img_{idx:03d}.png")
            cv2.imwrite(path, frame)  # 원본(코너 표시 없는) 사진 저장
            hull = cv2.convexHull(corners.astype(np.int32))
            layer = np.zeros_like(coverage); cv2.fillConvexPoly(layer, hull, (0, 180, 0))
            coverage = cv2.max(coverage, layer)
            history.append((path, layer))
            idx += 1
            print(f"저장 {idx}장: {path}")
        if k == ord("u") and history:
            path, _ = history.pop()
            os.remove(path); idx -= 1
            coverage = np.zeros_like(coverage)
            for _, layer in history:
                coverage = cv2.max(coverage, layer)
            print(f"취소: {path}")

    cap.release(); cv2.destroyAllWindows()
    print(f"총 {idx}장 저장: {out}")


if __name__ == "__main__":
    main()
