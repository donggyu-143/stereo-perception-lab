#!/usr/bin/env python3
"""
EXP-18: 같은 사진에서 "전(탐지기 혼자) vs 후(2단계 + 합성 데이터)" 비교 그림
========================================================================
- 같은 탐지기 박스를 쓰고, 종류(30/50)만
    전: 탐지기가 낸 종류 (EXP-16 방식)
    후: 잘라낸 표지판을 합성 데이터로 학습한 분류기가 낸 종류 (EXP-18)
  로 비교 → 바뀐 것은 오직 "숫자 판단" 부분
- 정답 표지판마다 분류
    고침  : 전 틀림 → 후 맞음   (합성 데이터 효과)
    망가짐: 전 맞음 → 후 틀림   (부작용, 정직하게 같이 표시)
    둘 다 틀림 / 둘 다 맞음
- 저장
  1) figures/<name>/fixed_*.jpg, broke_*.jpg : 사진 전체를 왼쪽(전)·오른쪽(후)으로 나란히
  2) figures/<name>_sheet.jpg : 고침·망가짐 표지판만 잘라서 한 장에 모은 요약 그림

사용 예
  python code/before_after.py --det runs/detect/v4_seed1/weights/best.pt --cls runs/classify/B_s0/weights/best.pt \
      --images datasets/sign_test2_rf/all/images --name ba_test2
"""
import argparse, glob, os
import cv2
import numpy as np

from find_failures import load_gt, iou
from make_cls_dataset import crop_sign

GREEN, RED, GRAY = (0, 170, 0), (0, 0, 255), (120, 120, 120)


def draw(img, items, names, title):
    """items: [(box, 예측 클래스, 맞음여부)]"""
    vis = img.copy()
    for b, c, ok in items:
        col = GREEN if ok else RED
        x1, y1, x2, y2 = map(int, b[:4])
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 4)
        label = f"{names[c][-2:]} {'O' if ok else 'X'}"
        cv2.putText(vis, label, (x1, max(y1 - 10, 30)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6)
        cv2.putText(vis, label, (x1, max(y1 - 10, 30)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 50), (255, 255, 255), -1)
    cv2.putText(vis, title, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--cls", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--name", default="before_after")
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()

    from ultralytics import YOLO
    det, clf = YOLO(a.det), YOLO(a.cls)
    names = [clf.names[i] for i in sorted(clf.names)]
    img_dir = os.path.expanduser(a.images)
    lab_dir = os.path.expanduser(a.labels) if a.labels else os.path.join(os.path.dirname(img_dir), "labels")
    fig_root = os.path.expanduser("~/calibration_lab/figures")
    out_dir = os.path.join(fig_root, a.name)
    os.makedirs(out_dir, exist_ok=True)

    count = {"fixed": 0, "broke": 0, "both_ok": 0, "both_wrong": 0, "missed": 0}
    sheet = []
    for p in sorted(q for e in ("jpg", "jpeg", "png") for q in glob.glob(os.path.join(img_dir, f"*.{e}"))):
        img = cv2.imread(p)
        h, w = img.shape[:2]
        gts = load_gt(os.path.join(lab_dir, os.path.splitext(os.path.basename(p))[0] + ".txt"), w, h)
        if not gts:
            continue
        r = det.predict(img, conf=a.conf, agnostic_nms=True, verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        before = r.boxes.cls.cpu().numpy().astype(int)
        after = [int(x.probs.top1) for x in clf.predict([crop_sign(img, b) for b in boxes], imgsz=128, verbose=False)] if len(boxes) else []

        items_b, items_a, kinds = [], [], set()
        for g in gts:
            j = max(range(len(boxes)), key=lambda k: iou(boxes[k], g), default=None)
            if j is None or iou(boxes[j], g) < 0.5:
                count["missed"] += 1
                continue
            t = g[5]
            ok_b, ok_a = before[j] == t, after[j] == t
            kind = "fixed" if (not ok_b and ok_a) else "broke" if (ok_b and not ok_a) else \
                   "both_ok" if ok_b else "both_wrong"
            count[kind] += 1
            kinds.add(kind)
            items_b.append((boxes[j], before[j], ok_b))
            items_a.append((boxes[j], after[j], ok_a))
            if kind in ("fixed", "broke", "both_wrong"):
                c = crop_sign(img, boxes[j], clahe=False)          # 요약 그림은 원본 밝기 그대로
                c = cv2.resize(c, (160, 160))
                c = cv2.copyMakeBorder(c, 0, 44, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                cv2.putText(c, f"true {names[t][-2:]}", (4, 176), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
                cv2.putText(c, f"before {names[before[j]][-2:]} {'O' if ok_b else 'X'}", (4, 192),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN if ok_b else RED, 1)
                cv2.putText(c, f"after {names[after[j]][-2:]} {'O' if ok_a else 'X'}", (84, 192),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN if ok_a else RED, 1)
                col = {"fixed": GREEN, "broke": RED, "both_wrong": GRAY}[kind]
                cv2.rectangle(c, (0, 0), (159, 159), col, 4)
                sheet.append((kind, c))

        for kind in ("fixed", "broke"):
            if kind in kinds:
                pair = np.hstack([draw(img, items_b, names, "BEFORE: detector only (EXP-16)"),
                                  np.full((h, 12, 3), 255, np.uint8),
                                  draw(img, items_a, names, "AFTER: 2-stage + synthetic (EXP-18)")])
                pair = cv2.resize(pair, None, fx=0.5, fy=0.5)
                cv2.imwrite(os.path.join(out_dir, f"{kind}_{os.path.basename(p)}"), pair)

    order = {"fixed": 0, "broke": 1, "both_wrong": 2}
    sheet.sort(key=lambda x: order[x[0]])
    if sheet:
        tiles = [c for _, c in sheet]
        cols = 6
        while len(tiles) % cols:
            tiles.append(np.full_like(tiles[0], 230))
        grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
        cv2.imwrite(os.path.join(fig_root, f"{a.name}_sheet.jpg"), grid)

    print(f"고침(전 X → 후 O) {count['fixed']} / 망가짐(전 O → 후 X) {count['broke']} / "
          f"둘 다 틀림 {count['both_wrong']} / 둘 다 맞음 {count['both_ok']} / 탐지 못함 {count['missed']}")
    print(f"나란히 그림: {out_dir}/fixed_*.jpg, broke_*.jpg")
    print(f"요약 그림: {os.path.join(fig_root, a.name + '_sheet.jpg')}  (초록 테두리 = 고침, 빨강 = 망가짐, 회색 = 둘 다 틀림)")


if __name__ == "__main__":
    main()
