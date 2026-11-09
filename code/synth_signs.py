#!/usr/bin/env python3
"""
EXP-18: 합성 데이터로 숫자 분류용 표지판 이미지 만들기
====================================================
왜
- EXP-17: 분류기만 따로 봐도 정답률 약 75% → 병목은 구조가 아니라 데이터
  (특히 학습 범위보다 어두운 50 을 30 으로 읽음)
- 표지판 디자인을 정확히 알고 있으므로(make_signs.py 와 같은 디자인·글꼴), 코드로 그려서
  "아주 어둡고 노이즈 많은 50/30"을 촬영 없이 원하는 만큼 만든다

만드는 과정 (한 장마다 무작위로)
1. 깨끗한 표지판 그리기: 빨간 테두리 + 흰 바탕 + 검은 숫자 (DejaVu Sans Bold, make_signs.py 와 같은 글꼴)
2. 기울이기: 평면 회전 ±20°, 비스듬한 각도(가로로 눌림 0.6~1.0), 약간의 원근
3. 배경 위에 붙이기: 실제로 찍은 배경 사진(neg_*.jpg) 조각, 없으면 무작위 무늬
   - crop_sign 과 같은 구도(표지판이 사각형의 약 87%)로 맞춤
4. 멀리서 찍힌 효과: 작게 줄였다가 다시 키움 (22~128px → 뭉개짐, 사람이 못 읽을 만큼은 안 함)
5. 조명: 전체 밝기 0.09~1.2배, 일부는 역광(배경만 밝고 표지판은 어둡게), 색 틀어짐 ±15%
6. 노이즈: 어두울수록 강하게 (웹캠의 어두운 곳 잡음 흉내)
7. 흐림 + JPEG 압축 손상
8. 마지막에 crop_sign 과 똑같이 CLAHE + 128x128 (진짜 데이터와 같은 처리)

주의 (sim-to-real gap)
- 합성은 현실과 다를 수 있으므로 진짜 데이터와 섞어서 학습하고, 평가는 반드시 진짜 사진(test, test2)으로만 한다

사용 예
  # 미리보기 (32장 격자 그림)
  python code/synth_signs.py --preview --bg datasets/sign_v1/images
  # 진짜 분류 데이터 복사 후 train 에 합성 추가 (종류별 1500장)
  cp -r datasets/cls_v4 datasets/cls_v4_synth
  python code/synth_signs.py --out datasets/cls_v4_synth/train --n 1500 --bg datasets/sign_v1/images
"""
import argparse, glob, os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

NAMES = {30: "speed_30", 50: "speed_50"}
RED = (32, 25, 215)          # BGR, make_signs.py 의 #d71920
FONT_PATHS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"]
_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))


def load_font(size):
    for p in FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    import matplotlib.font_manager as fm      # 경로가 다르면 matplotlib 으로 찾기
    return ImageFont.truetype(fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold")), size)


def render_sign(num, D=256):
    """깨끗한 표지판 (BGR) + 알파 마스크. make_signs.py 와 같은 비율: 흰 안쪽 = 반지름의 78%"""
    img = Image.new("RGB", (D, D), (255, 255, 255))
    mask = Image.new("L", (D, D), 0)
    d, m = ImageDraw.Draw(img), ImageDraw.Draw(mask)
    r = D / 2 - 2
    d.ellipse([D / 2 - r, D / 2 - r, D / 2 + r, D / 2 + r], fill=(215, 25, 32))
    m.ellipse([D / 2 - r, D / 2 - r, D / 2 + r, D / 2 + r], fill=255)
    ri = r * 0.78
    d.ellipse([D / 2 - ri, D / 2 - ri, D / 2 + ri, D / 2 + ri], fill=(255, 255, 255))
    font = load_font(int(D * 0.36))
    text = str(num)
    bb = d.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    d.text((D / 2 - tw / 2 - bb[0], D / 2 - th / 2 - bb[1]), text, fill=(17, 17, 17), font=font)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), np.array(mask)


def bg_patch(bgs, rng, size):
    """배경 조각: 실제 배경 사진에서 무작위로 잘라냄, 없으면 무작위 색 + 무늬"""
    if bgs:
        im = bgs[rng.integers(len(bgs))]
        h, w = im.shape[:2]
        s = int(rng.uniform(0.1, 0.5) * min(h, w))
        y, x = rng.integers(0, h - s), rng.integers(0, w - s)
        return cv2.resize(im[y:y + s, x:x + s], (size, size))
    base = rng.uniform(40, 220, 3)
    patch = np.ones((size, size, 3)) * base + rng.normal(0, 25, (size, size, 3))
    return np.clip(cv2.GaussianBlur(patch, (0, 0), 6), 0, 255).astype(np.uint8)


def synth_one(num, rng, bgs, out=128):
    sign, mask = render_sign(num)
    D = sign.shape[0]
    # 2) 기울이기: 회전 + 가로 눌림(비스듬히 본 효과) + 약간의 원근
    ang = rng.uniform(-20, 20)
    sx = rng.uniform(0.6, 1.0)
    src = np.float32([[0, 0], [D, 0], [D, D], [0, D]])
    j = D * 0.05
    dst = np.float32([[0, 0], [D, 0], [D, D], [0, D]]) + rng.uniform(-j, j, (4, 2)).astype(np.float32)
    dst[:, 0] = (dst[:, 0] - D / 2) * sx + D / 2
    Hm = cv2.getPerspectiveTransform(src, dst)
    R = np.vstack([cv2.getRotationMatrix2D((D / 2, D / 2), ang, 1.0), [0, 0, 1]])
    M = R @ Hm
    sign = cv2.warpPerspective(sign, M, (D, D), borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(mask, M, (D, D)).astype(np.float32) / 255

    # 3) 배경 위에 붙이기 (crop_sign 구도: 표지판 지름 ≈ 사각형의 1/1.15, 위치 약간 흔들기)
    S = int(D * 1.15)
    canvas = bg_patch(bgs, rng, S).astype(np.float32)
    ox, oy = (S - D) // 2 + rng.integers(-8, 9), (S - D) // 2 + rng.integers(-8, 9)
    ox, oy = int(np.clip(ox, 0, S - D)), int(np.clip(oy, 0, S - D))
    a = mask[..., None]
    region = canvas[oy:oy + D, ox:ox + D]
    # 5) 조명: 표지판과 배경 밝기를 따로 (역광 = 배경 밝고 표지판 어두움)
    k_sign = rng.choice([rng.uniform(0.09, 0.3), rng.uniform(0.3, 1.2)], p=[0.5, 0.5])
    k_bg = k_sign * (rng.uniform(1.5, 4.0) if rng.random() < 0.25 else rng.uniform(0.8, 1.2))
    tint = rng.uniform(0.85, 1.15, 3)
    canvas *= k_bg * tint
    canvas[oy:oy + D, ox:ox + D] = region * k_bg * tint * (1 - a) + sign.astype(np.float32) * k_sign * tint * a
    canvas = np.clip(canvas, 0, 255)

    # 4) 멀리서 찍힌 효과: 작게 줄였다가 키우기
    # 너무 작으면 사람도 못 읽는 사진이 됨(EXP-16 라벨 잡음 교훈) → 최소 22px
    small = int(rng.uniform(22, 128))
    img = cv2.resize(canvas.astype(np.uint8), (small, small), interpolation=cv2.INTER_AREA)
    img = cv2.resize(img, (out, out), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    # 6) 노이즈: 어두울수록 강하게
    sigma = float(np.clip(6 / max(k_sign, 0.05), 3, 25)) * rng.uniform(0.5, 1.0)
    img += rng.normal(0, sigma, img.shape)
    # 7) 흐림 + JPEG
    if rng.random() < 0.5 and small > 40:   # 이미 작게 뭉갠 경우엔 흐림을 겹치지 않음 (읽을 수 없는 사진 방지)
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.2))
    img = np.clip(img, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(30, 95))])
    img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    # 8) 진짜 데이터와 같은 마지막 처리: CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = _CLAHE.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def load_bgs(folder):
    if not folder:
        return []
    paths = sorted(glob.glob(os.path.join(os.path.expanduser(folder), "*neg_*.jpg")))
    return [cv2.imread(p) for p in paths]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="저장할 train 폴더 (그 안에 speed_30, speed_50 폴더 생성)")
    ap.add_argument("--n", type=int, default=1500, help="종류별 장수")
    ap.add_argument("--bg", default=None, help="배경 사진 폴더 (neg_*.jpg 사용)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", action="store_true", help="32장 미리보기 그림만 저장")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    bgs = load_bgs(a.bg)
    print(f"배경 사진 {len(bgs)}장 사용" + ("" if bgs else " (없음 → 무작위 무늬)"))

    if a.preview:
        tiles = []
        for i in range(32):
            num = 30 if i % 2 == 0 else 50
            t = synth_one(num, rng, bgs)
            t = cv2.copyMakeBorder(t, 0, 18, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(t, NAMES[num], (4, 141), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            tiles.append(t)
        rows = [np.hstack(tiles[r * 8:(r + 1) * 8]) for r in range(4)]
        path = os.path.expanduser("~/calibration_lab/figures/synth_preview.jpg")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, np.vstack(rows))
        print(f"미리보기: {path}")
        return

    out = os.path.expanduser(a.out)
    for num, name in NAMES.items():
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        for i in range(a.n):
            cv2.imwrite(os.path.join(d, f"synth_{num}_{i:05d}.png"), synth_one(num, rng, bgs))
    print(f"합성 저장: {out}/speed_30, speed_50 각 {a.n}장")


if __name__ == "__main__":
    main()
