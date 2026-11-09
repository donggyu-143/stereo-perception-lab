#!/usr/bin/env python3
"""
EXP-18: 분류 데이터를 눈으로 확인하는 격자 그림 만들기
====================================================
- 분류용 폴더(폴더 이름 = 클래스)에서 사진을 골라 한 장의 격자 그림으로 저장
- --model 을 주면 그 분류 모델의 예측도 칸 아래에 표시 (초록 = 맞음, 빨강 = 틀림)
- 쓰임
  1) 합성 데이터가 진짜 사진과 비슷한지, 사람이 읽을 수 있는지 확인
  2) 진짜 test2 의 잘라낸 표지판에서 모델이 무엇을 맞히고 틀렸는지 확인

사용 예
  # 진짜 test2 + 모델 예측
  python code/cls_sheet.py --folder datasets/cls_v4_synth/test2 --model runs/classify/B_s0/weights/best.pt --name sheet_test2
  # 합성 train 무작위 48장
  python code/cls_sheet.py --folder datasets/cls_v4_synth/train --only synth --n 48 --name sheet_synth
  # 진짜 train 무작위 48장 (합성과 나란히 비교용)
  python code/cls_sheet.py --folder datasets/cls_v4_synth/train --only real --n 48 --name sheet_real
"""
import argparse, glob, os
import cv2
import numpy as np

CELL = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help=".../train 또는 .../test2 (안에 speed_30, speed_50 폴더)")
    ap.add_argument("--model", default=None, help="분류 모델 best.pt (주면 예측 표시)")
    ap.add_argument("--only", choices=["all", "real", "synth"], default="all", help="합성(synth_*)만 / 진짜만 / 모두")
    ap.add_argument("--n", type=int, default=0, help="무작위로 고를 장수 (0 = 전부)")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--name", default="sheet")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    folder = os.path.expanduser(a.folder)
    items = []
    for cls in sorted(os.listdir(folder)):
        for p in sorted(glob.glob(os.path.join(folder, cls, "*.png"))):
            is_syn = os.path.basename(p).startswith("synth_")
            if a.only == "all" or (a.only == "synth") == is_syn:
                items.append((p, cls))
    if a.n and len(items) > a.n:
        idx = np.random.default_rng(a.seed).choice(len(items), a.n, replace=False)
        items = [items[i] for i in sorted(idx)]
    if not items:
        raise SystemExit("사진이 없습니다 (폴더·--only 확인)")

    preds = [None] * len(items)
    if a.model:
        from ultralytics import YOLO
        clf = YOLO(a.model)
        res = clf.predict([p for p, _ in items], imgsz=CELL, verbose=False)
        preds = [(clf.names[int(r.probs.top1)], float(r.probs.top1conf)) for r in res]

    tiles, wrong = [], 0
    for (p, cls), pr in zip(items, preds):
        t = cv2.resize(cv2.imread(p), (CELL, CELL))
        t = cv2.copyMakeBorder(t, 0, 34, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(t, f"true {cls[-2:]}", (4, CELL + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
        if pr:
            ok = pr[0] == cls
            wrong += not ok
            cv2.putText(t, f"pred {pr[0][-2:]} {pr[1]:.2f}", (4, CELL + 29), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (0, 150, 0) if ok else (0, 0, 255), 1)
            if not ok:
                cv2.rectangle(t, (0, 0), (CELL - 1, CELL - 1), (0, 0, 255), 3)
        tiles.append(t)
    blank = np.full_like(tiles[0], 230)
    while len(tiles) % a.cols:
        tiles.append(blank)
    sheet = np.vstack([np.hstack(tiles[r:r + a.cols]) for r in range(0, len(tiles), a.cols)])
    out = os.path.join(os.path.expanduser("~/calibration_lab/figures"), f"{a.name}.jpg")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, sheet)
    print(f"{len(items)}장" + (f", 틀림 {wrong}장 (빨간 테두리)" if a.model else "") + f" → {out}")


if __name__ == "__main__":
    main()
