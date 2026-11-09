#!/usr/bin/env python3
"""
EXP-16: 촬영 조건이 클래스에 치우쳤는지 확인 (지름길 학습 의심 검증)
================================================================
배경
- 1차 학습 결과 test 에서 50 을 30 으로 착각한 경우가 많았고(14/27), 주로 역광·어두운 장면, 화면 끝에 잘린 장면이었다
- 가설: train 에서 "어두운 장면 / 잘린 장면" 이 주로 30 이었다면, 모델이 숫자 대신 그 조건을 단서로 배운다 (지름길 학습)
- 이 스크립트는 train(또는 다른 구간)의 박스마다 조건을 재서 클래스별로 비교한다

재는 것 (박스마다)
- 박스 안 밝기 (회색 평균 0~255): 어두울수록 역광·저조도
- 박스 안 밝기 / 사진 전체 밝기: 1 보다 많이 작으면 "배경보다 표지판이 어두움" = 역광
- 화면 끝에 닿음(잘림) 여부
- 박스 크기(화면 대비 %): 작을수록 멂

출력
- 클래스별로 각 조건 구간에 박스가 몇 개인지 표
- 한 구간에서 두 클래스 비율이 크게 다르면(예: 어두운 박스는 30 이 20개, 50 이 3개) 그 조건이 지름길 단서가 될 수 있다

사용 예
  python code/dataset_bias.py datasets/sign_v1_rf --split train
  python code/dataset_bias.py datasets/sign_v1_rf --split test
"""
import argparse, glob, os
import cv2
import numpy as np
import yaml


def bins(v, edges, names):
    for e, n in zip(edges, names):
        if v < e:
            return n
    return names[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--split", default="train")
    a = ap.parse_args()
    root = os.path.expanduser(a.root)
    with open(os.path.join(root, "data.yaml"), encoding="utf-8") as fp:
        d = yaml.safe_load(fp)
    names = d["names"] if isinstance(d["names"], list) else [d["names"][k] for k in sorted(d["names"])]

    # 조건별 구간 정의
    conds = {
        "박스 밝기": (lambda r: r["inside"], [60, 100, 140], ["~60 어두움", "60~100", "100~140", "140~ 밝음"]),
        "역광 정도 (박스/전체)": (lambda r: r["ratio"], [0.6, 0.85, 1.1], ["~0.6 강한 역광", "0.6~0.85", "0.85~1.1", "1.1~"]),
        "박스 크기 (화면 %)": (lambda r: r["area"], [1, 4, 15], ["~1% 멂", "1~4%", "4~15%", "15%~ 가까움"]),
        "화면 끝 잘림": (lambda r: r["edge"], [0.5], ["안 잘림", "잘림"]),
    }

    rows = []
    for img in sorted(p for e in ("jpg", "jpeg", "png") for p in glob.glob(os.path.join(root, a.split, "images", f"*.{e}"))):
        lab = os.path.join(root, a.split, "labels", os.path.splitext(os.path.basename(img))[0] + ".txt")
        if not os.path.exists(lab):
            continue
        g = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY)
        h, w = g.shape
        whole = g.mean()
        for line in open(lab):
            p = line.split()
            if len(p) < 5:
                continue
            c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:5])
            x1, y1 = int(max((cx - bw / 2) * w, 0)), int(max((cy - bh / 2) * h, 0))
            x2, y2 = int(min((cx + bw / 2) * w, w)), int(min((cy + bh / 2) * h, h))
            if x2 <= x1 or y2 <= y1:
                continue
            inside = g[y1:y2, x1:x2].mean()
            rows.append({"cls": c, "inside": inside, "ratio": inside / max(whole, 1),
                         "area": bw * bh * 100,
                         "edge": float(x1 <= 2 or y1 <= 2 or x2 >= w - 2 or y2 >= h - 2)})

    print(f"[{a.split}] 박스 {len(rows)}개  (" + ", ".join(f"{n} {sum(r['cls'] == i for r in rows)}" for i, n in enumerate(names)) + ")")
    for title, (f, edges, labels) in conds.items():
        print(f"\n{title}")
        print(f"{'구간':<16}" + "".join(f"{n:>12}" for n in names) + f"{'30 비율':>10}")
        for lb in labels:
            cnt = [sum(1 for r in rows if r["cls"] == i and bins(f(r), edges, labels) == lb) for i in range(len(names))]
            tot = sum(cnt)
            share = f"{cnt[0] / tot * 100:.0f}%" if tot else "-"
            print(f"{lb:<16}" + "".join(f"{c:>12}" for c in cnt) + f"{share:>10}")
    print("\n→ 한 구간의 '30 비율'이 전체 비율(약 50%)보다 크게 높으면, 그 조건이 30 쪽 지름길 단서일 수 있다")


if __name__ == "__main__":
    main()
