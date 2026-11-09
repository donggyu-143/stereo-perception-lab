#!/usr/bin/env python3
"""
EXP-17: 2단계 파이프라인(탐지 → 잘라내기 → 숫자 분류) 실제 정답률 평가
====================================================================
- [1단계] 탐지 모델(EXP-16 4차 seed 1)로 표지판 박스를 찾는다. 탐지기가 낸 종류(30/50)는 버리고 위치만 쓴다
  (종류 무관 NMS: 한 표지판에 박스 하나)
- [2단계] 박스를 make_cls_dataset.crop_sign 과 똑같이 잘라(정사각형·여유·CLAHE·128) 분류 모델로 30/50 결정
- 정답 라벨과 비교 (find_failures.py 와 같은 기준: IoU 0.5 이상 + 종류 일치 = 맞음)
  → EXP-16 결과(탐지기 혼자: 기존 test 69~83%, test2 73%)와 바로 비교 가능
- --compare 를 주면 같은 박스에서 "탐지기 혼자 낸 종류"의 결과도 같이 출력 (2단계 효과를 같은 조건에서 비교)

사용 예
  python code/two_stage_eval.py --det runs/detect/v4_seed1/weights/best.pt \
      --cls runs/classify/cls_v4/weights/best.pt \
      --images datasets/sign_test2_rf/all/images --name test2_2stage --compare
"""
import argparse, glob, os
import cv2
import numpy as np

from find_failures import load_gt, match
from make_cls_dataset import crop_sign


def summarize(title, tot, cm, names):
    P = tot["tp"] / max(tot["tp"] + tot["fp"] + tot["conf"], 1)
    R = tot["tp"] / max(tot["tp"] + tot["fn"] + tot["conf"], 1)
    n_gt = tot["tp"] + tot["conf"] + tot["fn"]
    print(f"\n[{title}] 정답 {tot['tp']}/{n_gt} ({tot['tp'] / max(n_gt, 1) * 100:.0f}%) | "
          f"혼동 {tot['conf']} / 오탐 {tot['fp']} / 놓침 {tot['fn']} | Precision {P:.3f}, Recall {R:.3f}")
    head = "정답 \\ 예측"
    print(f"{head:<14}" + "".join(f"{n:>10}" for n in names) + f"{'놓침':>8}")
    for t in range(len(names)):
        print(f"{names[t]:<14}" + "".join(f"{cm.get((t, p), 0):>10}" for p in range(len(names)))
              + f"{cm.get((t, 'miss'), 0):>8}")
    print(f"{'배경(오탐)':<14}" + "".join(f"{cm.get(('bg', p), 0):>10}" for p in range(len(names))))


def tally(preds, gts, tot, cm):
    tp, conf, fp, fn, _ = match(preds, gts)
    tot["tp"] += len(tp); tot["conf"] += len(conf); tot["fp"] += len(fp); tot["fn"] += len(fn)
    for p in tp: cm[(p[5], p[5])] = cm.get((p[5], p[5]), 0) + 1
    for p, g in conf: cm[(g[5], p[5])] = cm.get((g[5], p[5]), 0) + 1
    for g in fn: cm[(g[5], "miss")] = cm.get((g[5], "miss"), 0) + 1
    for p in fp: cm[("bg", p[5])] = cm.get(("bg", p[5]), 0) + 1
    return conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True, help="탐지 모델 best.pt")
    ap.add_argument("--cls", required=True, help="분류 모델 best.pt")
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--name", default="two_stage")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--cls-imgsz", type=int, default=128)
    ap.add_argument("--no-clahe", action="store_true", help="분류 데이터를 --no-clahe 로 만들었으면 같이 줌")
    ap.add_argument("--compare", action="store_true", help="같은 박스에서 탐지기 혼자 낸 종류 결과도 출력")
    a = ap.parse_args()

    from ultralytics import YOLO
    det, clf = YOLO(a.det), YOLO(a.cls)
    names = [clf.names[i] for i in sorted(clf.names)]
    img_dir = os.path.expanduser(a.images)
    lab_dir = os.path.expanduser(a.labels) if a.labels else os.path.join(os.path.dirname(img_dir), "labels")
    out_dir = os.path.join(os.path.expanduser("~/calibration_lab/figures"), f"{a.name}_failures")
    os.makedirs(out_dir, exist_ok=True)

    tot2 = {"tp": 0, "conf": 0, "fp": 0, "fn": 0}; cm2 = {}
    tot1 = {"tp": 0, "conf": 0, "fp": 0, "fn": 0}; cm1 = {}
    paths = sorted(p for e in ("jpg", "jpeg", "png") for p in glob.glob(os.path.join(img_dir, f"*.{e}")))
    for p in paths:
        img = cv2.imread(p)
        h, w = img.shape[:2]
        gts = load_gt(os.path.join(lab_dir, os.path.splitext(os.path.basename(p))[0] + ".txt"), w, h)
        r = det.predict(img, conf=a.conf, agnostic_nms=True, verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        dconf = r.boxes.conf.cpu().numpy()
        dcls = r.boxes.cls.cpu().numpy().astype(int)

        # 2단계: 박스마다 잘라서 분류
        preds2 = []
        if len(boxes):
            crops = [crop_sign(img, b, clahe=not a.no_clahe) for b in boxes]
            res = clf.predict(crops, imgsz=a.cls_imgsz, verbose=False)
            for b, c, rr in zip(boxes, dconf, res):
                preds2.append(list(b) + [float(c), int(rr.probs.top1)])
        conf2 = tally(preds2, gts, tot2, cm2)

        # 비교: 같은 박스에서 탐지기 혼자 낸 종류
        if a.compare:
            preds1 = [list(b) + [float(c), int(k)] for b, c, k in zip(boxes, dconf, dcls)]
            tally(preds1, gts, tot1, cm1)

        if conf2:   # 2단계에서도 틀린 사진은 그림으로 저장
            vis = img.copy()
            for q, g in conf2:
                cv2.rectangle(vis, tuple(map(int, q[:2])), tuple(map(int, q[2:4])), (255, 0, 255), 3)
                cv2.putText(vis, f"WRONG {names[q[5]]}", (int(q[0]), int(q[1]) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.imwrite(os.path.join(out_dir, os.path.basename(p)), vis)

    print(f"사진 {len(paths)}장, 탐지 확신도 기준 {a.conf}, 종류 무관 NMS")
    if a.compare:
        summarize("탐지기 혼자 (EXP-16 방식)", tot1, cm1, names)
    summarize("2단계: 탐지 → 잘라내기 → 숫자 분류", tot2, cm2, names)
    print(f"\n2단계에서도 틀린 사진: {out_dir}")


if __name__ == "__main__":
    main()
