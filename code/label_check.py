#!/usr/bin/env python3
"""
EXP-16: 라벨 실수 찾기 - 라벨 박스를 잘라서 클래스별로 한 장에 모아 보기
====================================================================
하는 일
- 지정한 번호 범위의 사진(예: 3차에 추가한 img_0241~0296)에서 라벨 박스를 모두 잘라낸다
- 잘라낸 표지판을 같은 크기로 키우고, 어두운 사진도 숫자가 보이게 밝기·대비를 자동으로 펴서(CLAHE) 보여준다
  (보기용 보정일 뿐, 데이터는 바꾸지 않음)
- 클래스별로 한 장씩 모아 저장: figures/label_check_speed_30.jpg, label_check_speed_50.jpg
  → "speed_30" 그림에 50 이 보이면(또는 반대면) 그 칸 아래 적힌 파일 이름이 라벨 실수

사용 예
  python code/label_check.py datasets/sign_v3_rf --start 241 --end 296
"""
import argparse, glob, os, re
import cv2
import numpy as np
import yaml

CELL = 160  # 칸 크기(px)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Roboflow export 폴더 (data.yaml 있는 곳)")
    ap.add_argument("--start", type=int, default=241)
    ap.add_argument("--end", type=int, default=296)
    ap.add_argument("--cols", type=int, default=8)
    a = ap.parse_args()
    root = os.path.expanduser(a.root)
    with open(os.path.join(root, "data.yaml"), encoding="utf-8") as fp:
        d = yaml.safe_load(fp)
    names = d["names"] if isinstance(d["names"], list) else [d["names"][k] for k in sorted(d["names"])]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))

    cells = {i: [] for i in range(len(names))}
    for split in ("train", "valid", "test"):
        for img_path in sorted(glob.glob(os.path.join(root, split, "images", "img_*"))):
            m = re.match(r"img_(\d+)", os.path.basename(img_path))
            if not m or not (a.start <= int(m.group(1)) <= a.end):
                continue
            lab = os.path.join(root, split, "labels", os.path.splitext(os.path.basename(img_path))[0] + ".txt")
            if not os.path.exists(lab):
                continue
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            for k, line in enumerate(open(lab)):
                p = line.split()
                if len(p) < 5:
                    continue
                c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:5])
                # 박스보다 10% 넓게 잘라서 테두리까지 보이게
                x1, y1 = int(max((cx - bw * 0.6) * w, 0)), int(max((cy - bh * 0.6) * h, 0))
                x2, y2 = int(min((cx + bw * 0.6) * w, w)), int(min((cy + bh * 0.6) * h, h))
                if x2 - x1 < 4 or y2 - y1 < 4:
                    continue
                crop = img[y1:y2, x1:x2]
                # 보기용 밝기 보정: 밝기 채널에만 CLAHE
                lab_img = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
                lab_img[:, :, 0] = clahe.apply(lab_img[:, :, 0])
                crop = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
                crop = cv2.resize(crop, (CELL, CELL - 20), interpolation=cv2.INTER_CUBIC)
                cell = np.full((CELL, CELL, 3), 255, np.uint8)
                cell[:CELL - 20] = crop
                cv2.putText(cell, f"{m.group(1)}-{k}", (4, CELL - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cells[c].append(cell)

    out_dir = os.path.expanduser("~/calibration_lab/figures")
    os.makedirs(out_dir, exist_ok=True)
    for c, lst in cells.items():
        if not lst:
            print(f"{names[c]}: 박스 없음"); continue
        rows = (len(lst) + a.cols - 1) // a.cols
        sheet = np.full((rows * CELL, a.cols * CELL, 3), 230, np.uint8)
        for i, cell in enumerate(lst):
            r, q = divmod(i, a.cols)
            sheet[r * CELL:(r + 1) * CELL, q * CELL:(q + 1) * CELL] = cell
        path = os.path.join(out_dir, f"label_check_{names[c]}.jpg")
        cv2.imwrite(path, sheet)
        print(f"{names[c]}: 박스 {len(lst)}개 → {path}")
    print("칸 아래 숫자 = 사진 번호-박스 순서 (예: 0253-1 = img_0253 의 두 번째 박스)")
    print("→ speed_30 그림에 50 이, speed_50 그림에 30 이 보이면 라벨 실수")


if __name__ == "__main__":
    main()
