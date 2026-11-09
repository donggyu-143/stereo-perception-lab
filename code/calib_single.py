#!/usr/bin/env python3
"""
1단계: 단일 카메라 캘리브레이션 계산 스크립트
=============================================
하는 일
1) captures/<name>/img_*.png 사진들에서 체커보드 코너를 찾는다
2) "코너의 실제 3D 좌표(mm)" 와 "사진 속 2D 좌표(px)" 쌍으로 calibrateCamera 를 돌려
   내부 파라미터(fx, fy, cx, cy)와 왜곡 계수(k1, k2, p1, p2, k3)를 구한다
3) 결과가 믿을 만한지 판단하는 보조 지표(표준편차, 촬영 범위, 보드 기울기)를 계산한다
4) 결과를 results/<name>.json 에 저장하고, results/results.csv 에 한 줄 추가한다 (실험끼리 비교용)
5) figures/ 에 그림 3장을 저장한다: 촬영 범위, 사진별 오차, 왜곡 보정 전후 비교

사용 예 (~/calibration_lab 에서 가상환경 켠 상태로)
  python code/calib_single.py --name EXP-2_camA_baseline --cam-label camA --note "기준: 15장"
  python code/calib_single.py --name EXP-2_camA_baseline --cam-label camA --fix-k3   ← k3를 0으로 고정해서 비교

전체 흐름에서 가장 중요한 세 부분 (면접 답변의 뼈대)
  [A] objp       : 보드 코너의 실제 3D 좌표 만들기
  [B] 반복문      : 사진마다 코너 찾기 + 서브픽셀로 다듬기
  [C] calibrate  : 3D-2D 쌍으로 카메라 파라미터 계산
"""
import argparse, csv, glob, json, os   # 한 줄에 여러 개 import 가능
#   csv  : 표 형식 파일(results.csv) 쓰기
#   json : 결과를 사람이 읽을 수 있는 텍스트 형식(.json)으로 저장
from datetime import datetime
import cv2
import numpy as np

# 실험 폴더의 최상위 경로 (모든 저장 위치의 기준)
ROOT = os.path.expanduser("~/calibration_lab")


def main():
    # ===============================================================
    # 0) 터미널 옵션
    # ===============================================================
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="captures 안의 실험 폴더 이름")
    ap.add_argument("--cam-label", default="", help="카메라 이름 (예: camA)")
    #   ※ 옵션 이름의 '-' 는 코드에서 '_' 로 바뀐다: --cam-label → a.cam_label
    ap.add_argument("--square", type=float, default=25.0, help="체커보드 칸 크기 (mm)")
    ap.add_argument("--cols", type=int, default=9, help="내부 코너 가로 개수")
    ap.add_argument("--rows", type=int, default=6, help="내부 코너 세로 개수")
    ap.add_argument("--note", default="", help="results.csv 에 남길 메모 (바꾼 조건 등)")
    # --fix-k3 를 붙이면 k3(r⁶ 항)를 0으로 고정하고 계산한다 (값을 주지 않는 on/off 옵션)
    #   action="store_true" : 옵션을 쓰면 True, 안 쓰면 False
    #   목적: k3 과적합 의심 확인. k3 없이도 오차가 비슷하면 더 단순한 모델이 더 믿을 만하다
    ap.add_argument("--fix-k3", action="store_true", help="왜곡 계수 k3 를 0으로 고정")
    a = ap.parse_args()

    # 결과 파일 이름표: k3 고정이면 뒤에 _fixk3 를 붙여서, 같은 사진으로 계산한 두 결과가 덮어써지지 않게 한다
    #   예: EXP-3_camA_angles.json / EXP-3_camA_angles_fixk3.json
    tag = a.name + ("_fixk3" if a.fix_k3 else "")

    pattern = (a.cols, a.rows)   # (9, 6)

    # 사진 목록 찾기: captures/<name>/img_*.png 패턴에 맞는 파일들을 이름 순서대로
    files = sorted(glob.glob(os.path.join(ROOT, "captures", a.name, "img_*.png")))
    if not files:   # 목록이 비어 있으면
        raise SystemExit(f"[오류] 사진이 없습니다: captures/{a.name}/img_*.png")

    # ===============================================================
    # [A] 보드 코너의 실제 3D 좌표 만들기  ← 캘리브레이션의 "정답지"
    # ===============================================================
    # 보드 위에 좌표계를 둔다: 왼쪽 위 코너 = (0, 0, 0), 보드 면 = Z=0 평면
    # 그러면 코너 좌표는 (0,0,0), (25,0,0), (50,0,0) ... (200,125,0) 처럼 25mm 간격으로 정해진다
    #
    # np.zeros((54, 3)) : 54행 3열(X, Y, Z) 배열을 0으로 만들고
    objp = np.zeros((a.cols * a.rows, 3), np.float32)
    # np.mgrid[0:9, 0:6] : (0,0), (1,0), ... (8,5) 같은 격자 번호를 만든다
    # .T.reshape(-1, 2)  : 모양을 54행 2열로 바꿈 (OpenCV가 코너를 돌려주는 순서와 맞추기 위함)
    # * a.square         : 칸 번호 × 25mm = 실제 위치(mm)
    # objp[:, :2] = ...  : 모든 행의 앞 두 열(X, Y)에 넣기. Z(세 번째 열)는 0 그대로
    objp[:, :2] = np.mgrid[0:a.cols, 0:a.rows].T.reshape(-1, 2) * a.square
    # → 결과의 단위가 mm 이므로, 나중에 나오는 보드 위치(tvecs)도 mm 단위가 된다

    # cornerSubPix 반복 종료 조건: "최대 30번 반복" 또는 "이동량이 0.001px 이하"가 되면 멈춤
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    obj_pts, img_pts, used = [], [], []   # 3D 좌표 목록, 2D 좌표 목록, 실제 사용한 사진 경로
    size = None                           # 이미지 크기 (가로, 세로) — 아래에서 채움

    # ===============================================================
    # [B] 사진마다 코너 찾기
    # ===============================================================
    for f in files:
        img = cv2.imread(f)                              # 사진 읽기 (BGR 컬러)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     # 흑백 변환
        size = gray.shape[::-1]
        #   gray.shape = (세로, 가로) = (720, 1280)
        #   [::-1] = 순서 뒤집기 → (1280, 720). OpenCV 함수들은 (가로, 세로) 순서를 원한다

        # 1차 검출: 코너 위치를 대략(픽셀 단위) 찾기
        #   (촬영 때와 달리 FAST_CHECK 를 빼서, 시간이 걸려도 꼼꼼하게 찾는다)
        ok, c = cv2.findChessboardCorners(gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ok:
            print(f"  코너 검출 실패, 제외: {os.path.basename(f)}")   # basename = 경로 빼고 파일 이름만
            continue   # 이 사진은 건너뛰고 다음 사진으로

        # 2차 정밀화(서브픽셀): 코너 주변 11x11 영역의 밝기 변화를 보고 소수점 위치까지 다듬는다
        #   예: (312, 205) → (312.47, 204.83)
        #   (-1, -1) = 가운데 제외 영역 없음
        #   이 단계 덕분에 재투영 오차를 0.1~0.3px 수준까지 낮출 수 있다
        c = cv2.cornerSubPix(gray, c, (11, 11), (-1, -1), crit)

        obj_pts.append(objp)   # 이 사진의 3D 좌표 (모든 사진에서 같은 보드라 매번 같은 objp)
        img_pts.append(c)      # 이 사진에서 찾은 2D 좌표
        used.append(f)

    print(f"사용한 사진: {len(used)} / {len(files)}장, 해상도 {size[0]}x{size[1]}")
    if len(used) < 5:
        raise SystemExit("[오류] 사용할 수 있는 사진이 너무 적습니다 (5장 이상 필요).")

    # ===============================================================
    # [C] 캘리브레이션 계산  ← 핵심 한 줄
    # ===============================================================
    # 입력: 3D 좌표 목록, 2D 좌표 목록, 이미지 크기, (초기값 없음 None, None)
    # 내부 동작: 파라미터를 추정 → 3D 코너를 사진 위로 "재투영" → 실제 코너와의 오차 계산
    #           → 오차가 줄어드는 방향으로 수정 ... 을 반복 (비선형 최소제곱, Levenberg–Marquardt)
    # 출력:
    #   rms      : 최종 재투영 오차 (픽셀)
    #   K        : 카메라 행렬 3x3 [[fx,0,cx],[0,fy,cy],[0,0,1]]
    #   dist     : 왜곡 계수 [k1, k2, p1, p2, k3]
    #   rvecs    : 사진마다 보드의 회전 (외부 파라미터) — 3개짜리 회전 벡터
    #   tvecs    : 사진마다 보드의 위치 (외부 파라미터, mm)
    #   std_in   : 내부 파라미터 각각의 표준편차 (얼마나 확신하는가)  ← Extended 버전에서만 나옴
    #   std_ex   : 외부 파라미터의 표준편차 (여기서는 안 씀)
    #   per_view : 사진별 재투영 오차                               ← Extended 버전에서만 나옴
    #
    # flags: 계산 방식 옵션. CALIB_FIX_K3 = k3 를 0으로 두고 나머지만 구한다 (0 = 옵션 없음)
    flags = cv2.CALIB_FIX_K3 if a.fix_k3 else 0
    rms, K, dist, rvecs, tvecs, std_in, std_ex, per_view = cv2.calibrateCameraExtended(
        obj_pts, img_pts, size, None, None, flags=flags)

    per_view = per_view.ravel()   # ravel = 여러 겹의 배열을 1줄로 펴기 → [0.17, 0.21, ...]
    std_in = std_in.ravel()       # 순서: fx, fy, cx, cy, k1, k2, p1, p2, k3, ...
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]   # K[행, 열] 로 꺼내기
    d = dist.ravel()

    # ===============================================================
    # 보조 지표: 결과를 "믿을 수 있는지" 판단하는 숫자들
    # ===============================================================
    # (1) 촬영 범위: 화면을 4x4 = 16칸으로 나눠, 코너가 한 번이라도 들어간 칸의 비율
    #     → 가장자리에 데이터가 없으면 왜곡 계수를 제대로 못 구한다
    allc = np.vstack([p.reshape(-1, 2) for p in img_pts])   # 모든 사진의 코너를 한 배열로 (N, 2)
    #   allc[:, 0] = 모든 x 좌표,  / size[0] * 4 → 0~4 사이 값,  int → 칸 번호 0~3
    #   np.clip(..., 0, 3) = 혹시 4가 나오면 3으로 자르기 (화면 오른쪽 끝 픽셀 대비)
    gx = np.clip((allc[:, 0] / size[0] * 4).astype(int), 0, 3)
    gy = np.clip((allc[:, 1] / size[1] * 4).astype(int), 0, 3)
    # zip(gx, gy) = (칸x, 칸y) 쌍 목록 → set 으로 중복 제거 → 서로 다른 칸 개수 ÷ 16
    coverage = len(set(zip(gx, gy))) / 16.0

    # (2) 보드 기울기: 각 사진에서 보드가 카메라 정면에서 몇 도 틀어져 있었는지
    #     → 기울기가 작으면(정면만) fx 와 거리가 구분되지 않아 fx 가 불확실해진다 (EXP-2의 핵심 교훈)
    tilts = []
    for r in rvecs:
        R, _ = cv2.Rodrigues(r)   # 회전 벡터(3개 숫자) → 회전 행렬(3x3)로 변환
        # R[2, 2] = 보드 면의 수직 방향(법선)과 카메라 광축(Z축)이 이루는 각의 cos 값
        # arccos → 각도(라디안), degrees → 도(°). abs 는 앞/뒤 방향 상관없이 보기 위함
        tilts.append(np.degrees(np.arccos(abs(R[2, 2]))))
    tilts = np.array(tilts)   # 리스트 → NumPy 배열 (평균·최댓값 계산을 쉽게)

    # (3) 화각: 가로 화각 = 2 * arctan( (가로 픽셀/2) / fx )
    hfov = 2 * np.degrees(np.arctan(size[0] / (2 * fx)))
    vfov = 2 * np.degrees(np.arctan(size[1] / (2 * fy)))

    # ===============================================================
    # 결과 출력
    # ===============================================================
    print(f"\n===== 결과 ({'k3 고정' if a.fix_k3 else '기본 모델'}) =====")      # \n = 줄바꿈
    print(f"재투영 오차 RMS : {rms:.4f} px")                     # :.4f = 소수점 넷째 자리까지
    # argmax = 가장 큰 값의 위치(번호) → 오차가 가장 큰 사진 이름 찾기
    print(f"사진별 오차      : 평균 {per_view.mean():.3f}, 최대 {per_view.max():.3f} px ({os.path.basename(used[int(per_view.argmax())])})")
    print(f"fx, fy           : {fx:.2f}, {fy:.2f}   (표준편차 ±{std_in[0]:.2f}, ±{std_in[1]:.2f})")
    print(f"cx, cy           : {cx:.2f}, {cy:.2f}   (표준편차 ±{std_in[2]:.2f}, ±{std_in[3]:.2f})  / 영상 중심 {size[0]/2:.0f}, {size[1]/2:.0f}")
    # ', '.join(...) = 여러 문자열을 ', ' 로 이어 붙이기. d[:5] = 앞에서 5개
    print(f"왜곡 k1,k2,p1,p2,k3: {', '.join(f'{v:.4f}' for v in d[:5])}")
    print(f"화각 (가로/세로) : {hfov:.1f}° / {vfov:.1f}°")
    print(f"촬영 범위        : 화면 4x4 칸 중 {coverage*100:.0f}%")
    print(f"보드 기울기      : 평균 {tilts.mean():.1f}°, 최대 {tilts.max():.1f}°")

    # ===============================================================
    # 저장 1: JSON (이 실험의 모든 결과) — 다음 단계(스테레오)에서 K, dist 를 읽어 쓸 파일
    # ===============================================================
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)

    # 딕셔너리(dict): {"이름": 값, ...} 형태로 결과를 묶음
    # .tolist() : NumPy 배열은 JSON 에 바로 못 넣어서 파이썬 기본 리스트로 변환
    res = {
        "name": tag, "fix_k3": a.fix_k3, "cam_label": a.cam_label, "time": datetime.now().isoformat(timespec="seconds"),
        "image_size": list(size), "pattern": list(pattern), "square_mm": a.square,
        "n_images": len(files), "n_used": len(used), "rms_px": rms,
        "K": K.tolist(), "dist": d.tolist(), "std_fx_fy_cx_cy": std_in[:4].tolist(),
        # 사진 이름: 오차 형태의 딕셔너리. zip 으로 두 목록을 짝지어 돌림
        "per_view_error_px": {os.path.basename(u): float(e) for u, e in zip(used, per_view)},
        "hfov_deg": hfov, "vfov_deg": vfov, "coverage_4x4": coverage,
        "tilt_mean_deg": float(tilts.mean()), "tilt_max_deg": float(tilts.max()), "note": a.note,
    }
    # with open(...) as fp: → 파일을 열고, 블록이 끝나면 자동으로 닫아 준다
    #   "w" = 쓰기 모드(덮어쓰기), encoding="utf-8" = 한글이 깨지지 않게
    with open(os.path.join(ROOT, "results", f"{tag}.json"), "w", encoding="utf-8") as fp:
        json.dump(res, fp, ensure_ascii=False, indent=2)   # ensure_ascii=False = 한글 그대로, indent=2 = 보기 좋게 들여쓰기

    # ===============================================================
    # 저장 2: CSV (실험마다 한 줄씩 누적) — 엑셀로 열어 EXP-2, EXP-3 ... 을 한눈에 비교
    # ===============================================================
    csv_path = os.path.join(ROOT, "results", "results.csv")
    new = not os.path.exists(csv_path)   # 파일이 처음 만들어지는지 확인 (그때만 제목 줄을 씀)
    # "a" = append(이어 쓰기) 모드: 기존 내용 뒤에 추가. newline="" = 빈 줄이 끼지 않게
    with open(csv_path, "a", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        if new:   # 제목 줄
            w.writerow(["time", "name", "cam", "stage", "n_used", "rms_px", "fx", "fy", "cx", "cy",
                        "std_fx", "std_cx", "k1", "k2", "p1", "p2", "k3", "hfov_deg",
                        "coverage_4x4", "tilt_mean_deg", "note"])
        # *[...] = 리스트를 풀어서 각각의 칸으로 넣기 (k1~k3 다섯 개)
        w.writerow([res["time"], tag, a.cam_label, "single", len(used), f"{rms:.4f}",
                    f"{fx:.2f}", f"{fy:.2f}", f"{cx:.2f}", f"{cy:.2f}", f"{std_in[0]:.2f}", f"{std_in[2]:.2f}",
                    *[f"{v:.5f}" for v in d[:5]], f"{hfov:.1f}", f"{coverage:.2f}", f"{tilts.mean():.1f}", a.note])

    # ===============================================================
    # 그림 1: 촬영 범위 — 모든 사진의 코너 위치를 흰 바탕에 초록 점으로
    # ===============================================================
    cov = np.full((size[1], size[0], 3), 255, np.uint8)   # 255로 채운 이미지 = 흰색
    # 4x4 칸을 나누는 회색 선 (가로선 3개, 세로선 3개)
    for i in range(1, 4):   # i = 1, 2, 3
        #   // = 정수 나눗셈 (소수점 버림)
        cv2.line(cov, (size[0] * i // 4, 0), (size[0] * i // 4, size[1]), (220, 220, 220), 1)
        cv2.line(cov, (0, size[1] * i // 4), (size[0], size[1] * i // 4), (220, 220, 220), 1)
    for p in img_pts:                       # 사진마다
        for x, y in p.reshape(-1, 2):       # 코너마다
            cv2.circle(cov, (int(x), int(y)), 3, (0, 140, 0), -1)   # 반지름 3 원, -1 = 속을 채움
    cv2.putText(cov, f"{tag}  coverage {coverage*100:.0f}%  ({len(used)} images)", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.imwrite(os.path.join(ROOT, "figures", f"{tag}_coverage.png"), cov)

    # ===============================================================
    # 그림 2: 사진별 재투영 오차 막대그래프 (matplotlib 없이 OpenCV로 직접 그림)
    # ===============================================================
    H, W = 360, max(600, 40 * len(used) + 80)    # 그림 크기: 사진 수에 맞게 가로 길이 조절
    bar = np.full((H, W, 3), 255, np.uint8)
    mx = max(per_view.max(), 1e-6)              # 가장 큰 오차 (0으로 나누기 방지용 아주 작은 값과 비교)
    # enumerate = (번호, 값) 을 같이 돌려줌 → i = 사진 번호, e = 그 사진의 오차
    for i, e in enumerate(per_view):
        x0, h = 60 + i * 40, int((e / mx) * (H - 90))   # 막대 x 위치, 막대 높이 (가장 큰 오차 = 최대 높이)
        # 이미지 좌표는 y가 아래로 갈수록 커지므로, 바닥(H-40)에서 위로 h 만큼 사각형을 그림
        cv2.rectangle(bar, (x0, H - 40 - h), (x0 + 28, H - 40), (60, 120, 220), -1)
        cv2.putText(bar, f"{i}", (x0 + 6, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)        # 아래: 사진 번호
        cv2.putText(bar, f"{e:.2f}", (x0 - 2, H - 45 - h), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1) # 위: 오차 값
    cv2.putText(bar, f"per-image reprojection error [px]  RMS {rms:.3f}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.imwrite(os.path.join(ROOT, "figures", f"{tag}_per_image_error.png"), bar)

    # ===============================================================
    # 그림 3: 왜곡 보정 전후 비교
    # ===============================================================
    # 왜곡은 가장자리에서 크게 보이므로, 보드가 화면 중심에서 가장 멀리 있던 사진을 고른다
    #   각 사진의 코너들과 화면 중심 사이 거리 중 최댓값 → 그게 가장 큰 사진의 번호
    far = int(np.argmax([np.abs(p.reshape(-1, 2) - [size[0] / 2, size[1] / 2]).max() for p in img_pts]))
    src = cv2.imread(used[far])

    # getOptimalNewCameraMatrix: 보정 후 이미지에 쓸 새 카메라 행렬
    #   alpha=1 → 원본의 모든 픽셀이 보이게 (대신 가장자리에 검은 영역이 생김)
    #   alpha=0 이면 검은 영역 없이 잘라냄. 여기서는 왜곡 모양을 보려고 1을 씀
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, size, 1, size)
    und = cv2.undistort(src, K, dist, None, newK)   # 왜곡 보정 실행

    # 두 이미지에 똑같이 주황 가로선 7개 + 라벨을 그림 → 직선이 직선으로 보이는지 비교
    #   for A, B in ((a1, b1), (a2, b2)) : 쌍을 하나씩 꺼내서 반복
    for im, label in ((src, "original"), (und, "undistorted")):
        for i in range(1, 8):
            cv2.line(im, (0, size[1] * i // 8), (size[0], size[1] * i // 8), (0, 200, 255), 1)
        cv2.putText(im, label, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
    cv2.imwrite(os.path.join(ROOT, "figures", f"{tag}_undistort.png"), np.hstack([src, und]))

    print(f"\n저장: results/{tag}.json, results/results.csv")
    print(f"그림: figures/{tag}_coverage.png, _per_image_error.png, _undistort.png")


if __name__ == "__main__":
    main()
