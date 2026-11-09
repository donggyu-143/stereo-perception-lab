#!/usr/bin/env python3
"""
EXP-17: 숫자 분류 모델(YOLO11n-cls) 학습 + 구간별 정답률
=====================================================
- make_cls_dataset.py 가 만든 폴더(train/val/test/test2, 폴더 이름 = 클래스)로 분류 모델을 학습
- 출발 모델: yolo11n-cls.pt (ImageNet 사전학습 분류 모델) → 2클래스(speed_30, speed_50)로 파인튜닝
- 학습 후 val / test / test2 각각에서 정답률과 혼동 표를 직접 계산해서 출력 (test2 는 ultralytics 가 모르는 폴더라 직접 계산)
- 이 단계 정답률은 "정답 박스를 잘라서 줬을 때 숫자를 맞히는가" = 분류 능력만 따로 본 값
  (실제 파이프라인 정답률은 two_stage_eval.py 에서 탐지기 박스로 다시 잰다)

실행 (Colab)
  !python train_cls.py --data cls_v4 --name cls_v4 --seed 0
"""
import argparse, glob, json, os


def eval_folder(model, folder, imgsz):
    """folder/클래스이름/*.png → 정답률, 혼동 표"""
    names = model.names                                   # {0: 'speed_30', 1: 'speed_50'}
    idx = {v: k for k, v in names.items()}
    cm = [[0] * len(names) for _ in names]                # cm[정답][예측]
    for cls_name in sorted(os.listdir(folder)):
        paths = sorted(glob.glob(os.path.join(folder, cls_name, "*.png")))
        if not paths or cls_name not in idx:
            continue
        for r in model.predict(paths, imgsz=imgsz, verbose=False, stream=True):
            cm[idx[cls_name]][int(r.probs.top1)] += 1
    total = sum(map(sum, cm))
    acc = sum(cm[i][i] for i in range(len(cm))) / max(total, 1)
    return acc, cm, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="make_cls_dataset.py 출력 폴더")
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="yolo11n-cls.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=128)
    ap.add_argument("--fliplr", type=float, default=0.5, help="좌우 반전 확률 (숫자는 뒤집으면 거울 글자 → 0 과 비교 가능)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from ultralytics import YOLO
    data = os.path.abspath(os.path.expanduser(a.data))
    model = YOLO(a.model)
    model.train(data=data, epochs=a.epochs, imgsz=a.imgsz, fliplr=a.fliplr, seed=a.seed,
                patience=15, name=a.name, exist_ok=True, plots=True)
    best = os.path.join(model.trainer.save_dir, "weights", "best.pt")
    clf = YOLO(best)

    out = {"best": best}
    print("\n================ 분류 결과 (정답 박스를 잘라 줬을 때) ================")
    for split in ("val", "test", "test2"):
        folder = os.path.join(data, split)
        if not os.path.isdir(folder):
            continue
        acc, cm, n = eval_folder(clf, folder, a.imgsz)
        out[split] = {"acc": acc, "n": n, "cm": cm}
        print(f"{split:<6} 정답률 {acc*100:5.1f}% ({sum(cm[i][i] for i in range(len(cm)))}/{n})  "
              f"30→50 {cm[0][1]}, 50→30 {cm[1][0]}")
    with open(os.path.join(model.trainer.save_dir, "summary_cls.json"), "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    print(f"best 모델: {best}")


if __name__ == "__main__":
    main()
