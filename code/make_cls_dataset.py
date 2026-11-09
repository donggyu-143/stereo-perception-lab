#!/usr/bin/env python3
"""
EXP-17: 2단계 구조용 숫자 분류 데이터셋 만들기 (라벨 박스 잘라내기)
=================================================================
왜 2단계인가
- EXP-16: YOLO 탐지는 표지판 위치를 거의 다 찾지만(놓침 0~1), 30/50 구분은 약 70%
- YOLO 는 화면 전체를 640 으로 줄여서 보므로 작은 표지판의 숫자가 몇 픽셀밖에 안 됨
  + 30/50 의 차이는 앞 숫자의 윗부분(5 의 가로선 vs 3 의 곡선)뿐
- 그래서 [1단계] 위치는 YOLO 로 찾고, [2단계] 표지판만 잘라 크게 확대한 뒤 숫자만 분류하는 모델을 따로 둔다
  (실제 교통 표지판 인식 시스템도 흔히 이 구조: 검출기 + 분류기)

하는 일
- 탐지용 데이터(YOLO 형식)의 정답 박스를 잘라서 분류용 폴더 구조로 저장
    out/train/speed_30/*.png, out/train/speed_50/*.png, out/val/..., out/test/..., out/test2/...
  (ultralytics 분류 학습이 이 폴더 구조를 그대로 읽음: 폴더 이름 = 클래스)
- 잘라낼 때 처리 (crop_sign 함수, 평가 때도 똑같이 사용해야 함 = 학습/사용 영상 일치)
  1) 박스를 정사각형으로 만들고 15% 여유를 둠 (테두리까지 보이게, 가로세로 비율 왜곡 방지)
  2) 대비 보정 CLAHE: 밝기 채널만, 국소적으로 대비를 펴서 어두운 표지판의 5 가로선을 살림
  3) 128x128 로 크기 통일

사용 예
  python code/make_cls_dataset.py --out datasets/cls_v4 \
      --train datasets/sign_v4_rf/train --val datasets/sign_v4_rf/valid \
      --test datasets/sign_v4_rf/test --test2 datasets/sign_test2_rf/all
"""
import argparse, glob, os
import cv2
import numpy as np

NAMES = ["speed_30", "speed_50"]          # data.yaml 순서와 같아야 함 (0 = 30, 1 = 50)
_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))


def crop_sign(img, box, margin=0.15, size=128, clahe=True):
    """img: BGR 원본, box: (x1, y1, x2, y2) 픽셀 → size x size BGR
    정사각형 + 여유 → 화면 밖은 가장자리 색으로 채움(BORDER_REPLICATE) → CLAHE → 크기 맞춤"""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * (1 + margin) / 2
    a, b = int(round(cx - half)), int(round(cy - half))
    s = max(int(round(half * 2)), 4)
    h, w = img.shape[:2]
    pad = s  # 화면 밖으로 나가도 잘라낼 수 있게 미리 테두리를 붙임
    big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    crop = big[b + pad:b + pad + s, a + pad:a + pad + s]
    if clahe:
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = _CLAHE.apply(lab[:, :, 0])
        crop = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA if s > size else cv2.INTER_CUBIC)


def yolo_boxes(label_path, w, h):
    """YOLO txt → [(클래스, (x1, y1, x2, y2))]"""
    out = []
    if not os.path.exists(label_path):
        return out
    for line in open(label_path):
        p = line.split()
        if len(p) < 5:
            continue
        c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:5])
        out.append((c, ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)))
    return out


def export_split(src, dst, no_clahe):
    """src: .../images 와 .../labels 가 있는 폴더"""
    cnt = [0, 0]
    for c in range(len(NAMES)):
        os.makedirs(os.path.join(dst, NAMES[c]), exist_ok=True)
    imgs = sorted(p for e in ("jpg", "jpeg", "png") for p in glob.glob(os.path.join(src, "images", f"*.{e}")))
    for ip in imgs:
        img = cv2.imread(ip)
        h, w = img.shape[:2]
        stem = os.path.splitext(os.path.basename(ip))[0]
        for k, (c, box) in enumerate(yolo_boxes(os.path.join(src, "labels", stem + ".txt"), w, h)):
            if box[2] - box[0] < 4 or box[3] - box[1] < 4:
                continue
            cv2.imwrite(os.path.join(dst, NAMES[c], f"{stem}_{k}.png"), crop_sign(img, box, clahe=not no_clahe))
            cnt[c] += 1
    return cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", default=None)
    ap.add_argument("--test2", default=None)
    ap.add_argument("--no-clahe", action="store_true", help="대비 보정 끄기 (효과 비교용)")
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    print(f"{'구간':<8}{'speed_30':>10}{'speed_50':>10}")
    for split, src in (("train", a.train), ("val", a.val), ("test", a.test), ("test2", a.test2)):
        if not src:
            continue
        c = export_split(os.path.expanduser(src), os.path.join(out, split), a.no_clahe)
        print(f"{split:<8}{c[0]:>10}{c[1]:>10}")
    print(f"저장: {out}  (Colab 에 올려 분류 학습: train_cls.py)")


if __name__ == "__main__":
    main()
