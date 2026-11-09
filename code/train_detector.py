#!/usr/bin/env python3
"""
EXP-16: 커스텀 데이터셋으로 YOLO11n 파인튜닝 + 평가
====================================================
하는 일
1. COCO 로 미리 학습된 yolo11n.pt 에서 출발해서 속도제한 표지판(30, 50) 데이터로 추가 학습 (파인튜닝)
   - 처음부터 학습하면 수천 장이 필요하지만, 이미 "물체 모양 보는 법"을 배운 모델에서 출발하면 수백 장으로 충분
2. 학습이 끝나면 val, test 각각에서 성능을 계산해서 표로 출력하고 json 으로 저장
3. 학습 전 기본 모델과의 비교는 find_failures.py --model yolo11n.pt --ignore-class 로 한다
   (COCO 모델은 속도제한 표지판 클래스가 없어서 클래스 번호로 비교하는 val 은 의미가 없음.
    find_failures 는 클래스 상관없이 "표지판 위치에 뭐라도 박스를 쳤는지"로 세므로 공정한 비교가 됨)

지표 뜻
- Precision(정밀도): 모델이 "30"이라고 한 것 중 진짜 30 비율 (클래스별 계산 후 평균) → 낮으면 오탐·혼동이 많음
- Recall(재현율): 진짜 30 중 모델이 "30"으로 찾은 비율 → 낮으면 놓치거나 50 으로 착각한 것이 많음
- mAP50: 박스가 정답과 50% 이상 겹치면 맞음으로 칠 때의 평균 정밀도 (가장 많이 보는 값)
- mAP50-95: 겹침 기준을 50%~95%로 바꿔가며 평균 → 박스 위치가 얼마나 정확한지까지 반영

클래스 2개일 때 더 볼 것
- 혼동행렬: 30 을 50 으로(또는 반대로) 착각한 횟수. 위치는 맞혀도 종류가 틀리면 둘 다 점수가 깎인다

데이터 준비 (Roboflow)
- 라벨링 후 Export → 형식 "YOLOv11" (또는 YOLOv8, 같은 형식) → zip 다운로드 → 압축 해제
- 폴더 안의 data.yaml 경로를 --data 로 준다
- 배경 사진(neg_*)은 라벨 없이 올리면 Roboflow 가 "null(빈 라벨)" 이미지로 포함해 준다

실행 위치
- Colab (GPU, 추천):
    !pip install ultralytics
    (zip 업로드 후) !unzip sign_v1.zip -d sign_v1
    !python resplit_by_name.py sign_v1        # 따로 찍은 test_ 사진만 test 로
    !python train_detector.py --data sign_v1/data.yaml --name sign_v1
- 노트북 CPU: 같은 명령. 200장·50 epoch 기준 1~2시간 정도 (--device cpu 자동)

사용 예
  python code/train_detector.py --data ~/calibration_lab/datasets/sign_v1_rf/data.yaml --name sign_v1
  결과: runs/detect/<name>/weights/best.pt  ← stereo_yolo.py --model 에 넣을 파일
"""
import argparse, json, os
import yaml


def metrics_row(m):
    """ultralytics 평가 결과에서 핵심 4개만 꺼낸다"""
    b = m.box
    return {"precision": float(b.mp), "recall": float(b.mr), "mAP50": float(b.map50), "mAP50-95": float(b.map)}


def has_split(data_yaml, split):
    """data.yaml 에 test 같은 구간이 정의돼 있는지 (Roboflow 기본 export 는 train/valid/test 모두 있음)"""
    with open(data_yaml, encoding="utf-8") as fp:
        d = yaml.safe_load(fp)
    return bool(d.get(split))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Roboflow export 의 data.yaml")
    ap.add_argument("--name", required=True, help="결과 폴더 이름 (예: sign_v1)")
    ap.add_argument("--model", default="yolo11n.pt", help="출발 모델 (COCO 사전학습)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640, help="학습 입력 크기. 먼 표지판의 숫자가 뭉개지면 960 도 시도")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fliplr", type=float, default=0.5,
                    help="좌우 반전 확률 (기본 0.5). 숫자 표지판은 뒤집으면 거울 글자 → 개선 단계에서 0 으로 비교")
    ap.add_argument("--hsv-v", type=float, default=0.4,
                    help="밝기 augmentation 폭 (기본 0.4 = 약 ±40%%). 키우면 학습 중 더 어둡거나 밝은 사진을 만들어 보여줌")
    ap.add_argument("--seed", type=int, default=0,
                    help="무작위 시드 (기본 0). 같은 설정을 시드만 바꿔 여러 번 돌리면 결과가 얼마나 흔들리는지 알 수 있음")
    ap.add_argument("--patience", type=int, default=20, help="val 성능이 이만큼 epoch 동안 안 좋아지면 조기 종료")
    a = ap.parse_args()

    from ultralytics import YOLO
    data = os.path.expanduser(a.data)
    split = "test" if has_split(data, "test") else "val"
    out = {"data": data, "split_for_final": split}

    # ---- 학습 ----
    # augmentation 은 ultralytics 기본값 사용 (밝기·색 변화, 좌우 반전, 크기/위치 변화, mosaic 등)
    # → 개선 단계에서 실패 원인에 따라 hsv_v(밝기), scale(크기) 등을 조정해 비교한다
    model = YOLO(a.model)
    model.train(data=data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                patience=a.patience, fliplr=a.fliplr, hsv_v=a.hsv_v, seed=a.seed, name=a.name, exist_ok=True, plots=True)
    best = os.path.join(model.trainer.save_dir, "weights", "best.pt")
    out["best"] = best

    # ---- 평가: 학습에 안 쓴 사진으로 ----
    tuned = YOLO(best)
    out["tuned_val"] = metrics_row(tuned.val(data=data, split="val", imgsz=a.imgsz, plots=False))
    out["tuned_" + split] = metrics_row(tuned.val(data=data, split=split, imgsz=a.imgsz, plots=True,
                                                   name=a.name + "_" + split, exist_ok=True))

    # ---- 출력 ----
    print("\n================ 결과 ================")
    print(f"best 모델: {best}")
    for k, v in out.items():
        if isinstance(v, dict):
            print(f"{k:>18}: P {v['precision']:.3f}  R {v['recall']:.3f}  "
                  f"mAP50 {v['mAP50']:.3f}  mAP50-95 {v['mAP50-95']:.3f}")
    gap = out["tuned_val"]["mAP50"] - out["tuned_" + split]["mAP50"] if split != "val" else 0
    if split != "val":
        print(f"val - {split} mAP50 차이: {gap:+.3f}  (크면 val 에 맞춰진 것 = 일반화 부족 의심)")
    with open(os.path.join(model.trainer.save_dir, "summary.json"), "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    print(f"저장: {os.path.join(model.trainer.save_dir, 'summary.json')}")
    print("학습 곡선: results.png / 혼동행렬(confusion_matrix.png)·PR곡선: 같은 폴더의 png 파일들")
    print("클래스별 P/R/mAP 는 위 val 출력의 클래스별 줄에서 확인 (30 과 50 중 어느 쪽이 약한지)")


if __name__ == "__main__":
    main()
