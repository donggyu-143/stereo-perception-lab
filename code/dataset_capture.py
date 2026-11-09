#!/usr/bin/env python3
"""
EXP-16: 커스텀 물체(속도제한 표지판 30/50) 탐지용 데이터셋 촬영 스크립트
=======================================================
하는 일
- 왼쪽 카메라 한 대만 열어서, 정렬(rectify)된 왼쪽 영상을 학습용 사진으로 저장한다
  → 나중에 stereo_yolo.py 가 "정렬된 왼쪽 영상"에서 탐지하므로, 학습 사진도 같은 모양(왜곡 보정·정렬 후)으로 맞춘다
    (학습 때와 사용 때 영상이 다르면 성능이 떨어진다 = 도메인 차이)
- 자동 모드: 일정 간격마다 자동 저장. 표지판을 옮기고 돌리면서 찍으면 200장도 금방 모인다
- 자동 저장 때 두 가지를 걸러낸다
  1) 흔들린 사진: 라플라시안 분산(선명도)이 기준보다 낮으면 저장 안 함
  2) 거의 같은 사진: 직전 저장 사진과 차이가 너무 작으면 저장 안 함 (같은 장면 수십 장은 학습에 도움이 안 됨)
- 배경 사진(표지판 없는 사진) 모드: 파일 이름이 neg_ 로 시작. 라벨 없이 학습에 넣으면 "표지판이 아닌 것"을 배운다 (오탐 감소)
- test 전용 촬영: --prefix test 로 찍으면 test_0001.jpg 로 저장 → resplit_by_name.py 가 이 사진들만 test 로 모은다
  (같은 자리에서 연속으로 찍은 사진을 무작위로 나누면 거의 같은 사진이 train 과 test 에 같이 들어가서
   test 점수가 실제보다 높게 나온다 = 데이터 누수. test 는 장소·조명·배경을 바꿔서 따로 찍는다)

저장 위치: ~/calibration_lab/datasets/<name>/images/img_0001.jpg, neg_0001.jpg (test 촬영은 test_0001.jpg, test_neg_0001.jpg)

키
  SPACE : 한 장 저장 (거르기 없이 바로)
  a     : 자동 모드 켜기/끄기
  n     : 배경 사진 모드 켜기/끄기 (표지판을 치우고 사용)
  q     : 종료

촬영 체크리스트 (화면에도 표시)
- 클래스 균형: 30 과 50 을 비슷한 장수로 (예: 각 100장). 둘이 같이 나오는 사진도 일부
- 거리: 50cm ~ 2m 이상까지 고르게 (먼 거리 = 작은 표지판도 꼭 포함)
- 각도: 정면만 찍지 말고 좌우로 30~45° 돌린 것, 카메라 높이를 바꾼 것, 화면 가장자리에 둔 것
- 조명: 형광등 / 스탠드만 / 창문 역광 / 어둡게
- 배경: 바닥, 책상, 이불, 벽 앞 등 여러 곳 (빨간 물건 근처도 → 오탐 확인)
- 가림: 손이나 물건으로 일부 가린 것
- 배경 사진(표지판 없음): 전체의 10% 정도 (200장이면 20장)
- test(30~40장): 다른 날·다른 장소·다른 조명에서 --prefix test 로 따로

사용 예
  python code/dataset_capture.py --cam 0 --stereo EXP-10_stereo --name sign_v1                # train/valid 용
  python code/dataset_capture.py --cam 0 --stereo EXP-10_stereo --name sign_v1 --prefix test  # test 용 (장소·조명 바꿔서)
  (정렬 없이 원본으로 찍으려면 --stereo 를 빼면 된다)
"""
import argparse, glob, os, re, time
import cv2
import numpy as np

ROOT = os.path.expanduser("~/calibration_lab")


def open_cam(i, w, h, fps):
    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise SystemExit(f"[오류] /dev/video{i} 를 열 수 없습니다.")
    return cap


def next_index(folder, prefix):
    """이미 저장된 파일 다음 번호부터 이어서 저장 (스크립트를 껐다 켜도 덮어쓰지 않게)"""
    nums = [int(m.group(1)) for f in glob.glob(os.path.join(folder, f"{prefix}_*.jpg"))
            if (m := re.search(rf"{prefix}_(\d+)\.jpg$", f))]
    return max(nums, default=0) + 1


def sharpness(gray):
    """라플라시안 분산: 경계가 또렷할수록 큼. 흔들리면 경계가 퍼져서 작아진다"""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def small(gray):
    """중복 비교용 작은 영상 (64x36). 작게 줄이면 노이즈 영향이 줄고 비교가 빠르다"""
    return cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="왼쪽 카메라 번호")
    ap.add_argument("--stereo", default=None, help="정렬 정보 이름 (예: EXP-10_stereo). 없으면 원본 저장")
    ap.add_argument("--name", required=True, help="데이터셋 폴더 이름 (예: sign_v1)")
    ap.add_argument("--prefix", default="img", help="파일 이름 앞부분. test 전용 촬영은 test")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--interval", type=float, default=0.8, help="자동 모드 저장 간격(초)")
    ap.add_argument("--min-sharp", type=float, default=60.0, help="자동 모드 선명도 기준 (화면 값 보고 조정)")
    ap.add_argument("--min-diff", type=float, default=6.0, help="자동 모드 중복 기준 (직전 사진과의 평균 밝기 차이)")
    ap.add_argument("--view-scale", type=float, default=0.6)
    a = ap.parse_args()

    out = os.path.join(ROOT, "datasets", a.name, "images")
    os.makedirs(out, exist_ok=True)

    # 정렬 맵: stereo_yolo.py 와 똑같은 함수로 만든다 (같은 정렬 = 같은 영상)
    mapL = None
    if a.stereo:
        from stereo_yolo import load_rectify
        maps = load_rectify(a.stereo)[0]
        mapL = maps[0]

    cap = open_cam(a.cam, a.width, a.height, a.fps)
    auto, neg = False, False
    last_t, last_small = 0.0, None
    P_IMG = a.prefix
    P_NEG = "neg" if a.prefix == "img" else f"{a.prefix}_neg"
    n_img, n_neg = next_index(out, P_IMG) - 1, next_index(out, P_NEG) - 1
    msg, msg_t = "", 0.0

    print(f"[저장 위치] {out}")
    print("SPACE 저장 / a 자동 / n 배경 모드 / q 종료")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[경고] 프레임을 못 읽음 (USB 재연결 필요할 수 있음)"); time.sleep(0.1); continue
        if mapL is not None:
            frame = cv2.remap(frame, mapL[0], mapL[1], cv2.INTER_LINEAR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sh = sharpness(gray)
        sm = small(gray)
        diff = np.inf if last_small is None else float(np.mean(np.abs(sm - last_small)))

        def save(reason):
            nonlocal n_img, n_neg, last_small, last_t, msg, msg_t
            if neg:
                n_neg += 1; path = os.path.join(out, f"{P_NEG}_{n_neg:04d}.jpg")
            else:
                n_img += 1; path = os.path.join(out, f"{P_IMG}_{n_img:04d}.jpg")
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            last_small, last_t = sm, time.time()
            msg, msg_t = f"saved {os.path.basename(path)} ({reason})", time.time()
            print(f"저장: {os.path.basename(path)} ({reason})")

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            save("manual")
        elif key == ord("a"):
            auto = not auto
        elif key == ord("n"):
            neg = not neg

        # 자동 저장: 간격이 지났고, 선명하고, 직전 사진과 충분히 다를 때만
        if auto and time.time() - last_t >= a.interval:
            if sh < a.min_sharp:
                msg, msg_t = "blurry -> skipped", time.time()
            elif diff < a.min_diff:
                msg, msg_t = "same as last -> move/rotate the sign", time.time()
            else:
                save("auto")

        # ---- 화면 표시 (저장되는 사진에는 안 들어감: 축소한 복사본 view 에만 그림) ----
        view = cv2.resize(frame, None, fx=a.view_scale, fy=a.view_scale)
        col = (0, 200, 255) if neg else (0, 255, 0)
        lines = [
            f"[{a.prefix}] sign {n_img}  neg {n_neg}  " + ("(target: 200 / ~20)" if a.prefix == "img" else "(target: 30~40 / ~4)"),
            f"AUTO {'ON' if auto else 'off'}   MODE {'BACKGROUND (no sign)' if neg else 'SIGN'}",
            f"sharp {sh:.0f} (min {a.min_sharp:.0f})   diff {diff if np.isfinite(diff) else 0:.1f} (min {a.min_diff:.0f})",
            "30:50 balance / distance / angle / light / background / both / occluded",
        ]
        for i, t in enumerate(lines):
            cv2.putText(view, t, (10, 25 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(view, t, (10, 25 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1)
        if msg and time.time() - msg_t < 1.0:      # 최근 1초 안의 저장/건너뜀 메시지 (OpenCV 글자는 영어만 됨)
            cv2.putText(view, msg, (10, view.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(view, msg, (10, view.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("dataset_capture", view)

    cap.release(); cv2.destroyAllWindows()
    print(f"[완료] 표지판 사진 {n_img}장, 배경 사진 {n_neg}장 → {out}")


if __name__ == "__main__":
    main()
