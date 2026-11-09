#!/usr/bin/env python3
# ↑ "shebang" 줄: 리눅스에서 이 파일을 직접 실행할 때 python3로 실행하라는 표시 (python 명령으로 실행하면 무시됨)
"""
0단계: 카메라 2대 동시 동작 확인 스크립트
==========================================
하는 일
- 두 카메라를 같은 해상도 / MJPG 포맷으로 열고, 두 영상을 옆으로 붙여서 한 창에 보여준다
- 실제로 적용된 해상도와 초당 프레임 수(FPS)를 화면과 터미널에 표시한다
- 키보드: s = 두 카메라 사진을 동시에 저장 / q = 종료

사용 예 (터미널에서, ~/calibration_lab 폴더에서 가상환경을 켠 상태로)
  python code/cam_check.py --left 2 --right 4
  python code/cam_check.py --left 2 --right 4 --width 1280 --height 720
  python code/cam_check.py --left 0 --right 6 --width 1280 --height 720 --view-scale 0.4   ← 화면이 작으면 더 줄이기

※ 큰따옴표 세 개(\"\"\")로 감싼 이 부분은 "docstring"이라고 하는 설명문이다. 실행에는 영향 없음.
"""

# ---------------------------------------------------------------
# import: 다른 사람이 만들어 둔 기능(라이브러리)을 불러오는 부분
# ---------------------------------------------------------------
import argparse   # 터미널에서 --left 2 같은 옵션을 받아서 해석해 주는 표준 라이브러리
import os         # 폴더 만들기, 경로 합치기 등 운영체제 관련 기능
import time       # 현재 시간 (FPS 계산, 잠깐 기다리기)
from datetime import datetime  # 날짜·시간을 "20261105_221530" 같은 문자열로 만들 때 사용
import cv2        # OpenCV: 카메라 열기, 영상 처리, 화면 표시
import numpy as np  # NumPy: 숫자 배열(=이미지) 계산. 관례적으로 np 라는 짧은 이름으로 부른다


# ---------------------------------------------------------------
# 함수 정의: 카메라 하나를 열고 설정하는 작업을 묶어 둠
#   def 함수이름(입력1, 입력2, ...):  →  아래 들여쓰기된 부분이 함수 내용
#   두 카메라에 같은 작업을 반복하니까 함수로 만들어 두고 두 번 호출한다
# ---------------------------------------------------------------
def open_cam(index, w, h):
    """index 번 카메라(/dev/video{index})를 w x h 해상도, MJPG 포맷으로 열어서 돌려준다."""

    # VideoCapture: 카메라를 여는 OpenCV 객체
    #   index      → /dev/videoN 의 N
    #   CAP_V4L2   → 리눅스 표준 카메라 인터페이스(V4L2)를 직접 쓰라는 뜻 (리눅스에서 가장 안정적)
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)

    # cap.set(설정항목, 값): 카메라에 원하는 설정을 요청
    # FOURCC = 영상 포맷을 나타내는 4글자 코드. "MJPG" = 카메라가 JPEG로 압축해서 보내는 방식
    #   → 무압축(YUYV)보다 USB 데이터 양이 훨씬 적어서, 두 카메라가 같은 USB 컨트롤러를 나눠 써도 버틸 수 있다
    #   ※ *"MJPG" 는 문자열을 글자 4개로 풀어서 넘기는 문법: VideoWriter_fourcc('M','J','P','G') 와 같음
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)    # 가로 해상도 요청
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)   # 세로 해상도 요청
    cap.set(cv2.CAP_PROP_FPS, 30)           # 초당 30장 요청

    # 카메라가 안 열렸으면 프로그램을 종료하고 안내 메시지 출력
    if not cap.isOpened():
        raise SystemExit(f"[오류] /dev/video{index} 를 열 수 없습니다. v4l2-ctl --list-devices 로 번호를 확인하세요.")
        # f"...{변수}..." = f-string. 문자열 안의 {} 자리에 변수 값을 넣어 준다

    # "요청"한 값이 실제로 적용됐는지 다시 읽어서 확인한다
    #   카메라가 지원하지 않는 해상도를 요청하면, 가장 가까운 지원 해상도로 바뀌어 있을 수 있기 때문
    #   cap.get()은 소수(float)를 돌려주므로 int()로 정수로 바꿈
    aw, ah = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 실제 적용된 포맷 코드를 다시 읽기. 숫자 하나로 돌아오므로 글자 4개로 되돌린다
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    #   숫자 안에 8비트(1바이트)씩 글자 4개가 들어 있다
    #   (fourcc >> 8*i) & 0xFF  →  i번째 바이트만 꺼내기 (>> 는 비트 밀기, & 0xFF 는 아래 8비트만 남기기)
    #   chr(숫자) → 그 숫자에 해당하는 글자,  "".join(...) → 글자들을 이어 붙여 문자열로
    fmt = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))

    print(f"/dev/video{index}: 요청 {w}x{h} → 실제 {aw}x{ah}, 포맷 {fmt}")
    return cap   # 열어 둔 카메라 객체를 함수 밖으로 돌려줌


# ---------------------------------------------------------------
# 메인 함수: 프로그램의 실제 흐름
# ---------------------------------------------------------------
def main():
    # ---- 1) 터미널 옵션 받기 ----
    ap = argparse.ArgumentParser()
    # add_argument("--이름", type=자료형, required=필수여부, default=기본값, help=설명)
    ap.add_argument("--left", type=int, required=True, help="왼쪽 카메라 /dev/videoN 의 N")
    ap.add_argument("--right", type=int, required=True, help="오른쪽 카메라 /dev/videoN 의 N")
    ap.add_argument("--width", type=int, default=640)    # 안 주면 640
    ap.add_argument("--height", type=int, default=480)   # 안 주면 480
    # expanduser: "~" 를 실제 홈 폴더 경로(/home/kdg9858)로 바꿔 줌
    ap.add_argument("--out", default=os.path.expanduser("~/calibration_lab/captures/EXP-1_cam_check"))
    # 화면 표시 배율: 1280x720 두 장을 붙이면 가로 2560px 이라 노트북 화면을 넘는다
    #   → 보여줄 때만 줄인다 (기본 0.5배). 저장하는 사진은 항상 원본 해상도 그대로
    ap.add_argument("--view-scale", type=float, default=0.5, help="화면 표시 배율 (저장 사진에는 영향 없음)")
    a = ap.parse_args()   # 실제로 해석 → a.left, a.right, a.width ... 로 꺼내 쓸 수 있음

    # ---- 2) 저장 폴더 만들고 카메라 두 대 열기 ----
    os.makedirs(a.out, exist_ok=True)   # 폴더가 없으면 만들고, 이미 있으면 그냥 넘어감
    capL, capR = open_cam(a.left, a.width, a.height), open_cam(a.right, a.width, a.height)

    # FPS 계산용 변수들
    #   t0 = 측정 시작 시각, n = 그동안 받은 프레임 수, fps = 마지막으로 계산한 FPS, saved = 저장 횟수
    t0, n, fps, saved = time.time(), 0, 0.0, 0
    print("s: 저장  q: 종료")

    # ---- 3) 무한 반복: 프레임 받기 → 화면에 표시 → 키 입력 확인 ----
    while True:
        # 카메라에서 사진을 받는 방법은 두 단계로 나눌 수 있다
        #   grab()     : 카메라에서 프레임을 "붙잡기"만 함 (빠름)
        #   retrieve() : 붙잡은 프레임을 이미지로 "풀어내기" (MJPG 압축 해제 등, 상대적으로 느림)
        # 두 카메라를 grab → grab 한 뒤 retrieve → retrieve 하면,
        # 촬영 시점 차이가 grab 한 번 시간 정도로 줄어든다 (나중에 스테레오에서 중요)
        okL, okR = capL.grab(), capR.grab()
        if not (okL and okR):
            # 둘 중 하나라도 실패하면 경고 출력 후 0.1초 쉬고 다음 반복으로 (continue)
            print("[경고] 프레임을 받지 못했습니다 (L=%s, R=%s)" % (okL, okR)); time.sleep(0.1); continue
        _, fL = capL.retrieve(); _, fR = capR.retrieve()
        # retrieve()는 (성공여부, 이미지) 두 개를 돌려준다. 성공여부는 안 쓰므로 _ 로 버림
        # fL, fR 은 NumPy 배열: 모양이 (세로, 가로, 3) → 3은 색 채널 B, G, R (OpenCV는 RGB가 아니라 BGR 순서)

        # FPS 계산: 1초가 지날 때마다 "그동안 받은 장수 ÷ 걸린 시간"
        n += 1
        if time.time() - t0 >= 1.0:
            fps, n, t0 = n / (time.time() - t0), 0, time.time()

        # 두 이미지를 가로로 붙이기 (hstack = horizontal stack). 세로 크기가 같아야 함
        view = np.hstack([fL, fR])

        # 화면에 글자 쓰기: putText(이미지, 글자, (x, y) 위치, 글꼴, 크기, (B, G, R) 색, 두께)
        cv2.putText(view, f"L /dev/video{a.left}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # 오른쪽 영상 글자는 왼쪽 영상 가로 크기만큼 옮겨서 씀. fL.shape[1] = 가로 픽셀 수
        cv2.putText(view, f"R /dev/video{a.right}", (fL.shape[1] + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # {fps:.1f} = 소수점 첫째 자리까지 표시
        cv2.putText(view, f"{fL.shape[1]}x{fL.shape[0]}  {fps:.1f} fps  saved {saved}", (10, view.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        # 표시용으로만 축소 (INTER_AREA = 줄일 때 화질이 좋은 보간 방법)
        shown = cv2.resize(view, None, fx=a.view_scale, fy=a.view_scale, interpolation=cv2.INTER_AREA)
        cv2.imshow("cam_check (s: save, q: quit)", shown)   # 창에 이미지 표시

        # waitKey(1): 1ms 동안 키 입력을 기다림. 이 호출이 있어야 창이 실제로 갱신된다
        # & 0xFF: 시스템에 따라 붙는 추가 비트를 제거하고 키 코드만 남김
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):     # ord("q") = 글자 q 의 숫자 코드
            break             # while 반복문 탈출 → 종료로
        if k == ord("s"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")   # 예: 20261105_221530 (파일 이름이 겹치지 않게)
            cv2.imwrite(os.path.join(a.out, f"{ts}_L.png"), fL)    # 왼쪽 사진
            cv2.imwrite(os.path.join(a.out, f"{ts}_R.png"), fR)    # 오른쪽 사진
            cv2.imwrite(os.path.join(a.out, f"{ts}_LR.png"), view) # 나란히 붙인 사진
            saved += 1
            print(f"저장: {a.out}/{ts}_*.png")

    # ---- 4) 정리: 카메라와 창 닫기 ----
    print(f"마지막 측정 FPS: {fps:.1f}")
    capL.release(); capR.release(); cv2.destroyAllWindows()


# 이 파일을 "직접 실행"했을 때만 main()을 호출한다
# (다른 파일에서 import 할 때는 실행되지 않게 하는 파이썬 관용구)
if __name__ == "__main__":
    main()
