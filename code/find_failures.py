#!/usr/bin/env python3
"""
EXP-16: 학습한 모델이 틀린 사진만 골라내서 원인 분석하기
======================================================
하는 일
- test(또는 valid) 사진마다 모델 예측과 정답 라벨을 비교한다
  - 예측 박스와 정답 박스가 IoU 0.5 이상 겹치고 종류도 같으면 "맞음"
  - 위치는 맞는데 종류가 틀림 = 혼동 (예: 50 을 30 이라고 함) ← 클래스가 2개 이상일 때 핵심
  - 정답인데 못 찾음 = 놓침(FN, miss)
  - 표지판이 아닌 곳에 박스 = 오탐(FP, false alarm)
- 틀린 사진만 그림으로 저장 (초록 = 정답, 보라 = 혼동, 빨강 = 오탐, 노랑 = 놓친 정답)
- 마지막에 "정답 종류 × 예측 종류" 혼동 표를 출력
- 원인 추정에 도움이 되는 수치도 같이 표로 출력
  - 박스 크기(화면 대비 %): 작으면 "너무 멂"
  - 사진 평균 밝기, 박스 안 vs 밖 밝기 차이: 어둡거나 박스 안이 훨씬 어두우면 "역광/저조도"
  - 박스가 화면 가장자리에 닿음: "잘림"
  → 최종 원인 분류(역광, 비슷한 물체, 가려짐 등)는 그림을 보고 직접 정한다

IoU(Intersection over Union): 두 박스 겹친 넓이 / 합친 넓이. 1이면 완전히 같음

라벨 형식 (YOLO txt): 한 줄에 "클래스 cx cy w h" (모두 0~1 비율)

학습 전/후 비교
- --model yolo11n.pt --ignore-class 로 같은 test 를 돌리면 "직접 학습 전"에 표지판을 얼마나 찾는지 나온다
  (COCO 모델은 클래스 번호가 다르므로 종류는 무시하고 박스 위치만 봄 → 다른 이름으로라도 잡았는지.
   COCO 에는 stop sign 만 있어서 속도제한 표지판을 그걸로 잡는지도 확인 가능)

사용 예
  python code/find_failures.py --model runs/detect/sign_v1/weights/best.pt \
      --images ~/calibration_lab/datasets/sign_v1_rf/test/images --name sign_v1
  python code/find_failures.py --model yolo11n.pt --ignore-class \
      --images ~/calibration_lab/datasets/sign_v1_rf/test/images --name sign_v1_base
"""
import argparse, glob, os
import cv2
import numpy as np


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt(label_path, w, h):
    """YOLO txt → 픽셀 좌표 박스 목록 [x1, y1, x2, y2, 1.0, 클래스]. 파일이 없거나 비면 배경 사진(표지판 없음)"""
    boxes = []
    if os.path.exists(label_path):
        for line in open(label_path):
            p = line.split()
            if len(p) < 5:
                continue
            cx, cy, bw, bh = (float(v) for v in p[1:5])
            boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h, 1.0, int(p[0])])
    return boxes


def match(preds, gts, thr=0.5, ignore_class=False):
    """박스 형식: [x1, y1, x2, y2, 확신도, 클래스]
    1) 확신도 높은 예측부터, 아직 짝 없는 "같은 종류" 정답과 IoU 최대로 짝짓기 (mAP 계산과 같은 방식)
    2) 남은 예측 중 남은 정답과 위치(IoU)는 맞는데 종류가 다른 것 = 혼동
    3) 그래도 남은 예측 중 "이미 맞힌 정답"과 많이 겹치는 것 = 중복(같은 표지판에 다른 종류 박스가 하나 더)
    4) 나머지 예측 = 오탐, 남은 정답 = 놓침
    반환: tp, conf(혼동 [(예측, 정답)]), fp(중복 포함), fn, dup(fp 중 중복인 것)"""
    def same(p, g):
        return ignore_class or p[5] == g[5]

    used, tp, rest = set(), [], []
    for p in sorted(preds, key=lambda x: -x[4]):
        best, bi = 0.0, -1
        for i, g in enumerate(gts):
            if i in used or not same(p, g):
                continue
            v = iou(p, g)
            if v > best:
                best, bi = v, i
        if best >= thr:
            used.add(bi); tp.append(p)
        else:
            rest.append(p)
    conf, fp, dup = [], [], []
    for p in rest:
        best, bi = 0.0, -1
        for i, g in enumerate(gts):
            if i in used:
                continue
            v = iou(p, g)
            if v > best:
                best, bi = v, i
        if best >= thr:
            used.add(bi); conf.append((p, gts[bi]))
        else:
            fp.append(p)
            if any(iou(p, g) >= thr for i, g in enumerate(gts) if i in used):
                dup.append(p)
    fn = [g for i, g in enumerate(gts) if i not in used]
    return tp, conf, fp, fn, dup


def box_info(img_gray, b):
    h, w = img_gray.shape
    x1, y1, x2, y2 = (int(round(v)) for v in b[:4])   # b 는 [x1, y1, x2, y2, 확신도, 클래스]
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
    area = (x2 - x1) * (y2 - y1) / (w * h) * 100
    inside = img_gray[y1:y2, x1:x2].mean() if x2 > x1 and y2 > y1 else np.nan
    edge = x1 <= 2 or y1 <= 2 or x2 >= w - 2 or y2 >= h - 2
    return area, inside, edge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True, help=".../test/images (labels 폴더가 옆에 있어야 함)")
    ap.add_argument("--labels", default=None, help="기본: images 옆의 labels 폴더")
    ap.add_argument("--name", default="run")
    ap.add_argument("--conf", type=float, default=0.25, help="예측으로 인정할 확신도 (stereo_yolo 와 맞춰 비교)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--ignore-class", action="store_true", help="종류 무시, 위치만 비교 (학습 전 COCO 모델 비교용)")
    ap.add_argument("--agnostic", action="store_true",
                    help="종류 무관 NMS: 같은 자리에 30·50 박스가 겹치면 확신도 높은 하나만 남김 (기본은 종류별로 따로 남김)")
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.model)
    img_dir = os.path.expanduser(a.images)
    lab_dir = os.path.expanduser(a.labels) if a.labels else os.path.join(os.path.dirname(img_dir), "labels")
    out_dir = os.path.join(os.path.expanduser("~/calibration_lab/figures"), f"{a.name}_failures")
    os.makedirs(out_dir, exist_ok=True)

    paths = sorted(p for ext in ("jpg", "jpeg", "png") for p in glob.glob(os.path.join(img_dir, f"*.{ext}")))
    tot = {"img": 0, "tp": 0, "conf": 0, "fp": 0, "fn": 0, "fail_img": 0}
    rows = []
    pn = model.names                                  # 예측 클래스 이름 (모델 기준)
    cm = {}                                           # 혼동 표: (정답 클래스, 예측 클래스 또는 "놓침") → 개수
    for p in paths:
        img = cv2.imread(p); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        gts = load_gt(os.path.join(lab_dir, os.path.splitext(os.path.basename(p))[0] + ".txt"), w, h)
        # NMS(겹친 박스 정리)는 기본적으로 "같은 종류끼리만" 한다 → 한 표지판에 speed_30 과 speed_50 박스가 둘 다 남을 수 있음
        # --agnostic 이면 종류와 상관없이 겹치면 확신도 높은 하나만 남긴다
        r = model.predict(img, conf=a.conf, imgsz=a.imgsz, agnostic_nms=a.agnostic, verbose=False)[0]
        preds = [list(b) + [float(c), int(k)] for b, c, k in
                 zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy())]
        tp, conf, fp, fn, dup = match(preds, gts, ignore_class=a.ignore_class)
        tot["dup"] = tot.get("dup", 0) + len(dup)
        tot["img"] += 1; tot["tp"] += len(tp); tot["conf"] += len(conf); tot["fp"] += len(fp); tot["fn"] += len(fn)
        if not a.ignore_class:
            for p_ in tp: cm[(p_[5], p_[5])] = cm.get((p_[5], p_[5]), 0) + 1
            for p_, g in conf: cm[(g[5], p_[5])] = cm.get((g[5], p_[5]), 0) + 1
            for g in fn: cm[(g[5], "miss")] = cm.get((g[5], "miss"), 0) + 1
            for p_ in fp: cm[("bg", p_[5])] = cm.get(("bg", p_[5]), 0) + 1
        if not conf and not fp and not fn:
            continue
        tot["fail_img"] += 1

        # ---- 그림 저장 ----
        vis = img.copy()
        for g in gts:
            cv2.rectangle(vis, tuple(map(int, g[:2])), tuple(map(int, g[2:4])), (0, 255, 0), 2)
        for g in fn:
            cv2.rectangle(vis, tuple(map(int, g[:2])), tuple(map(int, g[2:4])), (0, 255, 255), 3)
            cv2.putText(vis, "MISS", (int(g[0]), int(g[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        for q in fp:
            cv2.rectangle(vis, tuple(map(int, q[:2])), tuple(map(int, q[2:4])), (0, 0, 255), 2)
            cv2.putText(vis, f"FP {pn[q[5]]} {q[4]:.2f}", (int(q[0]), int(q[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        for q, g in conf:
            cv2.rectangle(vis, tuple(map(int, q[:2])), tuple(map(int, q[2:4])), (255, 0, 255), 3)
            cv2.putText(vis, f"WRONG {pn[q[5]]} {q[4]:.2f}", (int(q[0]), int(q[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(p)), vis)

        # ---- 원인 추정용 수치 ----
        mean_b = gray.mean()
        dup_ids = {id(q) for q in dup}
        for kind, lst in (("MISS", fn), ("DUP", [q for q in fp if id(q) in dup_ids]),
                          ("FP", [q for q in fp if id(q) not in dup_ids]), ("WRONG", [q for q, _ in conf])):
            for b in lst:
                area, inside, edge = box_info(gray, b)
                hint = []
                if area < 0.3: hint.append("작음(멂?)")
                if mean_b < 60: hint.append("어두움")
                if inside < mean_b - 40: hint.append("박스 안이 어두움(역광?)")
                if edge: hint.append("가장자리 잘림")
                if kind == "MISS" and len(gts) >= 2: hint.append("여러 개/가림?")
                if kind == "WRONG": hint.append("숫자 구분 실패 (멂·흐림·비스듬?)")
                rows.append((os.path.basename(p), kind, area, mean_b, inside, ", ".join(hint) or "-"))

    # ---- 표 출력 ----
    # 혼동은 "그 종류로는 틀림"이므로 Precision 에서는 오탐, Recall 에서는 놓침으로 센다 (mAP 계산과 같은 기준)
    P = tot["tp"] / max(tot["tp"] + tot["fp"] + tot["conf"], 1); R = tot["tp"] / max(tot["tp"] + tot["fn"] + tot["conf"], 1)
    print(f"\n사진 {tot['img']}장 중 틀린 사진 {tot['fail_img']}장 | 맞음 {tot['tp']} / 혼동 {tot['conf']} / 오탐 {tot['fp']} / 놓침 {tot['fn']}")
    print(f"conf {a.conf} 기준 Precision {P:.3f}, Recall {R:.3f}" + ("  (agnostic NMS)" if a.agnostic else ""))
    print(f"오탐 {tot['fp']}개 중 중복(이미 맞힌 표지판에 다른 종류 박스가 하나 더) {tot.get('dup', 0)}개 → --agnostic 으로 줄일 수 있음")
    print(f"\n{'사진':<28}{'종류':<6}{'박스%':>7}{'밝기':>7}{'박스안':>7}  추정 단서")
    for f, k, ar, mb, ins, hint in rows:
        print(f"{f[:27]:<28}{k:<6}{ar:7.2f}{mb:7.0f}{ins:7.0f}  {hint}")
    if cm:
        # 정답 클래스 번호는 data.yaml 순서, 예측 번호는 모델 순서 → 학습한 모델이면 둘이 같음
        keys_t = sorted({k[0] for k in cm}, key=str); keys_p = sorted({k[1] for k in cm}, key=str)
        name = lambda k: pn[k] if isinstance(k, int) else ("배경(정답 없음)" if k == "bg" else "놓침")
        print("\n혼동 표 (행 = 정답, 열 = 예측)")
        print(f"{'':<16}" + "".join(f"{name(k):>12}" for k in keys_p))
        for t in keys_t:
            print(f"{name(t):<16}" + "".join(f"{cm.get((t, k), 0):>12}" for k in keys_p))
    print(f"\n그림: {out_dir}  (초록 정답 / 노랑 MISS / 빨강 FP / 보라 WRONG = 종류 혼동)")
    print("→ 그림을 보고 원인을 직접 분류: 멂 / 역광·어두움 / 비스듬함 / 비슷한 물체 / 가려짐 / 잘림 / 라벨 실수")


if __name__ == "__main__":
    main()
