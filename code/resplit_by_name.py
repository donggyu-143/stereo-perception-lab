#!/usr/bin/env python3
"""
EXP-16: Roboflow export 를 "따로 찍은 test" 기준으로 다시 나누기 + 데이터셋 통계
================================================================================
왜 필요한가
- Roboflow 는 사진을 무작위로 train/valid/test 에 나눈다
- 같은 자리에서 연속으로 찍은 사진은 거의 똑같아서, 무작위로 나누면 "거의 같은 사진"이 train 과 test 에 함께 들어간다
  → 모델이 외운 장면으로 시험을 보는 셈이라 test 점수가 실제보다 높게 나온다 (데이터 누수)
- 그래서 test 는 장소·조명·배경을 바꿔 따로 찍고(dataset_capture.py --prefix test), 이 스크립트로 그 사진들만 test 에 모은다

하는 일
1. 파일 이름이 test_ 로 시작하는 사진(+라벨)은 모두 test/ 로 이동
2. test/ 안에 있던 나머지 사진은 train/ 으로 이동
3. data.yaml 의 train/val/test 경로를 이 폴더 기준으로 다시 씀
4. 구간별 사진 수, 클래스별 박스 수, 배경 사진 수를 표로 출력 (클래스 균형 확인)
   (원본 export 는 건드리기 싫으면 폴더를 먼저 복사해서 사용)

Roboflow 는 파일 이름을 test_0001_jpg.rf.<해시>.jpg 처럼 바꾸지만 앞부분(test_)은 유지된다

사용 예
  python code/resplit_by_name.py ~/calibration_lab/datasets/sign_v1_rf
"""
import argparse, glob, os, shutil
import yaml

SPLITS = ("train", "valid", "test")


def label_of(img_path):
    d = os.path.dirname(os.path.dirname(img_path))
    return os.path.join(d, "labels", os.path.splitext(os.path.basename(img_path))[0] + ".txt")


def move(img, dst_split, root):
    for sub, src in (("images", img), ("labels", label_of(img))):
        if os.path.exists(src):
            dst = os.path.join(root, dst_split, sub)
            os.makedirs(dst, exist_ok=True)
            shutil.move(src, os.path.join(dst, os.path.basename(src)))


def images(root, split):
    return sorted(p for ext in ("jpg", "jpeg", "png") for p in glob.glob(os.path.join(root, split, "images", f"*.{ext}")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Roboflow export 를 압축 푼 폴더 (data.yaml 이 있는 곳)")
    ap.add_argument("--test-prefix", default="test_")
    a = ap.parse_args()
    root = os.path.abspath(os.path.expanduser(a.root))

    moved_in = moved_out = 0
    for split in ("train", "valid"):
        for img in images(root, split):
            if os.path.basename(img).startswith(a.test_prefix):
                move(img, "test", root); moved_in += 1
    for img in images(root, "test"):
        if not os.path.basename(img).startswith(a.test_prefix):
            move(img, "train", root); moved_out += 1
    print(f"test 로 옮김 {moved_in}장 / test 에서 train 으로 옮김 {moved_out}장")

    # data.yaml 다시 쓰기 (절대 경로로 → Colab·노트북 어디서 실행해도 같은 폴더를 가리킴)
    ypath = os.path.join(root, "data.yaml")
    with open(ypath, encoding="utf-8") as fp:
        d = yaml.safe_load(fp)
    d["path"] = root
    d["train"], d["val"], d["test"] = "train/images", "valid/images", "test/images"
    with open(ypath, "w", encoding="utf-8") as fp:
        yaml.safe_dump(d, fp, allow_unicode=True, sort_keys=False)
    names = d["names"] if isinstance(d["names"], list) else [d["names"][k] for k in sorted(d["names"])]

    # 통계
    print(f"\n{'구간':<7}{'사진':>6}{'배경':>6}" + "".join(f"{n:>10}" for n in names))
    for split in SPLITS:
        imgs = images(root, split)
        cnt = [0] * len(names); bg = 0
        for img in imgs:
            lp = label_of(img)
            lines = [l.split() for l in open(lp)] if os.path.exists(lp) else []
            lines = [l for l in lines if len(l) >= 5]
            if not lines:
                bg += 1
            for l in lines:
                cnt[int(l[0])] += 1
        print(f"{split:<7}{len(imgs):>6}{bg:>6}" + "".join(f"{c:>10}" for c in cnt))
    print(f"\ndata.yaml 갱신: {ypath}")
    print("→ 클래스별 박스 수가 크게 차이 나면(예: 2배 이상) 적은 쪽을 더 찍는다")


if __name__ == "__main__":
    main()
