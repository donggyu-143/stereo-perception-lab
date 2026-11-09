#!/usr/bin/env python3
"""
1단계 보충: 교차 검증 (cross-validation) 스크립트
================================================
질문: "캘리브레이션 결과(내부 파라미터)가, 계산에 쓰지 않은 사진에서도 맞는가?"

왜 필요한가
- calib_single.py 가 출력하는 RMS 는 "계산에 쓴 사진으로 채점한 점수"다 (시험 문제를 보고 공부한 뒤 같은 문제로 시험)
- 게다가 실험마다 사진이 달라서 RMS 를 서로 비교할 수 없다 (시험지가 다름)
- 그래서 파라미터와 사진을 엇갈려서 채점한다 → 같은 사진(시험지)으로 여러 파라미터를 공정하게 비교

방법
- 파라미터 A (fx, fy, cx, cy, 왜곡) 를 "고정"한다 → 다시 계산하지 않는다
- 사진 세트 B 의 각 사진에서 코너를 찾고, 그 사진에서의 보드 위치·자세만 새로 구한다 (solvePnP)
- 그 자세 + 고정한 파라미터로 코너를 재투영 → 실제 코너와의 오차(RMS)를 잰다
  → 보드 자세는 사진마다 자유롭게 맞춰 주므로, 남는 오차는 "내부 파라미터가 틀려서" 생기는 오차다

사용 예 (~/calibration_lab 에서 가상환경 켠 상태로)
  python code/calib_crossval.py \
      --params EXP-2_camA_baseline EXP-2_camA_baseline_fixk3 EXP-3_camA_angles EXP-3_camA_angles_fixk3 \
      --images EXP-2_camA_baseline EXP-3_camA_angles \
      --name EXP-4_camA_crossval

  --params : results/ 안의 json 이름들 (calib_single.py 가 만든 결과, .json 은 빼고)
  --images : captures/ 안의 사진 폴더 이름들
결과: 표를 화면에 출력 + results/<name>.csv, results/<name>.json 저장
"""
import argparse, csv, glob, json, os
import cv2
import numpy as np

ROOT = os.path.expanduser("~/calibration_lab")


def load_params(name):
    """results/<name>.json 에서 K(카메라 행렬)와 dist(왜곡 계수)를 읽어 NumPy 배열로 돌려준다."""
    path = os.path.join(ROOT, "results", f"{name}.json")
    if not os.path.exists(path):
        raise SystemExit(f"[오류] 결과 파일이 없습니다: results/{name}.json (calib_single.py 를 먼저 실행)")
    with open(path, encoding="utf-8") as fp:
        r = json.load(fp)                        # json → 파이썬 딕셔너리
    K = np.array(r["K"], dtype=np.float64)       # 3x3 행렬
    dist = np.array(r["dist"], dtype=np.float64) # [k1, k2, p1, p2, k3]
    return K, dist, r


def detect_corners(folder, pattern, square):
    """captures/<folder>/img_*.png 에서 코너를 찾아 (3D 좌표, 2D 좌표, 파일 이름) 목록을 돌려준다.
    calib_single.py 와 '똑같은 방법'으로 찾아야 공정한 비교가 된다."""
    files = sorted(glob.glob(os.path.join(ROOT, "captures", folder, "img_*.png")))
    if not files:
        raise SystemExit(f"[오류] 사진이 없습니다: captures/{folder}/img_*.png")

    # 보드 코너의 실제 3D 좌표 (calib_single.py 의 [A] 와 동일)
    cols, rows = pattern
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    out = []
    for f in files:
        gray = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
        ok, c = cv2.findChessboardCorners(gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ok:
            continue
        c = cv2.cornerSubPix(gray, c, (11, 11), (-1, -1), crit)   # 서브픽셀 정밀화 (동일 조건)
        out.append((objp, c, os.path.basename(f)))
    return out


def evaluate(K, dist, samples):
    """고정된 K, dist 로 각 사진의 보드 자세만 구한 뒤 재투영 오차를 계산한다.
    돌려주는 값: 전체 RMS, 사진별 RMS 목록"""
    sq_sum, n_pts, per_img = 0.0, 0, []
    for objp, c, fname in samples:
        # solvePnP: "3D 점들과 그 점들이 찍힌 2D 위치, 그리고 카메라 파라미터"를 주면
        #           카메라에 대한 보드의 회전(rvec)과 위치(tvec)를 구해 준다 (= 외부 파라미터만 계산)
        #           내부 파라미터 K, dist 는 입력으로 주기만 하고 바꾸지 않는다 ← 교차 검증의 핵심
        ok, rvec, tvec = cv2.solvePnP(objp, c, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        # 구한 자세 + 고정 파라미터로 3D 코너를 사진 위에 다시 투영
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        # 실제 코너와 투영한 코너의 차이 (54개 점 각각의 x, y 차이)
        diff = (proj.reshape(-1, 2) - c.reshape(-1, 2))
        e2 = (diff ** 2).sum(axis=1)             # 점마다 거리의 제곱 (dx² + dy²)
        sq_sum += e2.sum(); n_pts += len(e2)
        per_img.append((fname, float(np.sqrt(e2.mean()))))
    # 전체 RMS = √(모든 점의 거리 제곱 평균) → calibrateCamera 가 출력하는 RMS 와 같은 정의
    return float(np.sqrt(sq_sum / n_pts)), per_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", nargs="+", required=True, help="results/ 의 json 이름들 (여러 개)")
    #   nargs="+" : 값을 1개 이상 여러 개 받는다 → 리스트가 됨
    ap.add_argument("--images", nargs="+", required=True, help="captures/ 의 사진 폴더 이름들 (여러 개)")
    ap.add_argument("--name", default="EXP-4_crossval", help="결과 파일 이름")
    ap.add_argument("--square", type=float, default=25.0)
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--rows", type=int, default=6)
    a = ap.parse_args()
    pattern = (a.cols, a.rows)

    # 1) 사진 세트마다 코너를 한 번만 찾아 둔다 (파라미터마다 다시 찾을 필요 없음)
    #    딕셔너리 컴프리헨션: {키: 값 for ...} → {"EXP-2_camA_baseline": [...], ...}
    sets = {img: detect_corners(img, pattern, a.square) for img in a.images}
    for img, s in sets.items():
        print(f"사진 세트 {img}: {len(s)}장")

    # 2) 모든 (파라미터, 사진 세트) 조합을 평가
    table = {}      # table[파라미터 이름][사진 세트 이름] = RMS
    detail = {}     # 사진별 오차 (json 저장용)
    for p in a.params:
        K, dist, r = load_params(p)
        table[p], detail[p] = {}, {}
        for img, samples in sets.items():
            rms, per = evaluate(K, dist, samples)
            table[p][img] = rms
            detail[p][img] = {"rms_px": rms, "per_image": dict(per),
                              "max_px": max(e for _, e in per), "max_image": max(per, key=lambda x: x[1])[0]}

    # 3) 표 출력 (행 = 파라미터, 열 = 채점에 쓴 사진 세트)
    print("\n===== 교차 검증: 재투영 오차 RMS [px] =====")
    print("행 = 어떤 파라미터로 / 열 = 어떤 사진으로 채점했는지")
    w = max(len(p) for p in a.params) + 2
    header = " " * w + "".join(f"{img:>26}" for img in a.images)
    print(header)
    for p in a.params:
        row = f"{p:<{w}}"
        for img in a.images:
            # 파라미터 이름이 사진 폴더 이름으로 시작하면 "자기 사진" = 계산에 쓴 사진 → * 표시
            own = "*" if p.startswith(img) else " "
            row += f"{table[p][img]:>25.4f}{own}"
        print(row)
    print("* = 계산에 쓴 사진으로 채점 (calib_single.py 의 RMS 와 거의 같아야 정상)")

    # 사진별 최대 오차도 보여 주기 (특정 사진에서만 크게 틀리는지)
    print("\n사진별 최대 오차 [px] (어느 사진에서 가장 크게 틀렸나)")
    for p in a.params:
        s = "  ".join(f"{img}: {detail[p][img]['max_px']:.3f} ({detail[p][img]['max_image']})" for img in a.images)
        print(f"  {p}: {s}")

    # 4) 저장
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", f"{a.name}.csv"), "w", newline="", encoding="utf-8") as fp:
        wr = csv.writer(fp)
        wr.writerow(["params \\ images"] + a.images)
        for p in a.params:
            wr.writerow([p] + [f"{table[p][img]:.4f}" for img in a.images])
    with open(os.path.join(ROOT, "results", f"{a.name}.json"), "w", encoding="utf-8") as fp:
        json.dump({"table_rms_px": table, "detail": detail}, fp, ensure_ascii=False, indent=2)
    print(f"\n저장: results/{a.name}.csv, results/{a.name}.json")


if __name__ == "__main__":
    main()
