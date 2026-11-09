#!/usr/bin/env python3
"""
6단계(EXP-15): YOLO 물체 검출 + 스테레오 조밀 시차로 물체 거리 재기
===================================================================
"무엇이(YOLO), 어디에, 얼마나 멀리(스테레오)" 있는지 카메라 두 대만으로 구한다.
자율주행 인지의 기본 구조인 "객체 검출 + 거리 추정"을 작게 재현한 것.

처리 흐름 (한 프레임)
1. 좌/우 카메라를 최대한 같은 순간에 찍기 (grab → grab → retrieve)
2. 정렬(rectify): 같은 점이 좌우 영상의 같은 가로줄에 오게 (3단계 결과 사용)
3. YOLO: 정렬된 왼쪽 컬러 영상에서 물체 검출 → 종류 + 박스(x1, y1, x2, y2) + 확신도
4. SGBM: 정렬된 좌우 회색 영상으로 조밀 시차 (EXP-13 결론: 1/4 축소, 노출 고정)
5. 박스마다 거리:
   - 박스 가운데 50% 영역만 사용 (박스 가장자리에는 배경이 섞이므로)
   - 그 안의 유효한 시차들의 "중앙값" d → Z = f·B / d
     (중앙값을 쓰는 이유: 배경·오답 시차가 일부 섞여도 다수인 물체 시차를 고른다)
   - 채움(fill): 영역 중 시차가 나온 비율. 낮으면 믿기 어려운 값 (무늬 없는 물체 등)
6. (추가) 박스 통째 매칭: 박스 영역 전체를 오른쪽 영상의 같은 가로줄에서 찾아 시차 하나 → 무늬가 적은 물체용
7. 화면 표시: "cup D 0.82m (95%)" = 조밀 방식(채움), "P 0.81m (0.92)" = 박스 매칭(상관 점수)
   박스 색: 초록 = 조밀 성공(채움 50% 이상) / 주황 = 조밀 실패지만 박스 매칭 성공(점수 0.6 이상) / 빨강 = 둘 다 실패

모드
- 실시간 (기본): 카메라 두 대로 실시간 표시. SPACE = 원본 L/R 저장, q = 종료
- 폴더 분석 (--folder): 저장한 사진 쌍을 다시 계산해 표로 출력. --tape 를 주면 줄자와 비교

주의
- 시차 범위 기본 96~576px → 약 34cm ~ 2m. 왼쪽 영상의 왼쪽 576px 띠는 시차 계산 불가(오른쪽 카메라에 안 보이는 영역)
  → 물체는 두 카메라 가운데 정면에 두면 된다
- 두 웹캠은 하드웨어 동기화가 없음 → 움직이는 물체는 거리 오차가 커질 수 있음 (측정은 정지 물체로)
- 거리 Z 는 "정렬된 왼쪽 카메라의 광축 방향 깊이". 카메라가 기울어 있으면 줄자(바닥 방향) 거리와 조금 다르다 (EXP-12 참고)
- YOLO 가 모르는 물체(COCO 80종 밖)는 --box x1 y1 x2 y2 로 직접 박스를 지정해 거리만 잴 수 있다 (폴더 모드)

사용 예
  python code/stereo_yolo.py --left 0 --right 6 --stereo EXP-10_stereo --name EXP-15_live
  python code/stereo_yolo.py --stereo EXP-10_stereo --folder EXP-15_yolo --tape 40 60 80 100 120 --cls bottle

표지판(EXP-16 확장) 옵션
- --model 직접 학습한 탐지기 best.pt → 종류 무관 NMS 자동 적용 (한 표지판 = 박스 하나)
- --cls-model 분류기 best.pt → 박스를 잘라(crop_sign, EXP-17과 같은 방식) 30/50을 분류기가 다시 판단 (2단계 구조)
- --size-mm 120 → 물체 실제 크기를 알 때 단안 거리 Zm = fx·H/w (박스 가로 w px) 도 같이 계산해 스테레오와 비교
  python code/stereo_yolo.py --left 4 --right 6 --stereo EXP-10_stereo --model det.pt --cls-model cls.pt --size-mm 120 --name EXP-16_sign_dist
  python code/stereo_yolo.py --stereo EXP-10_stereo --folder EXP-16_sign_dist --model det.pt --cls-model cls.pt --size-mm 120 --tape 50 70 90 110 130 150
"""
import argparse, glob, json, os, time
import cv2
import numpy as np

ROOT = os.path.expanduser("~/calibration_lab")


# ---------------- 준비: 캘리브레이션·정렬 정보 ----------------
def load_rectify(stereo):
    with open(os.path.join(ROOT, "results", f"{stereo}.json"), encoding="utf-8") as fp:
        s = json.load(fp)
    with open(os.path.join(ROOT, "results", f"{stereo}_rect.json"), encoding="utf-8") as fp:
        r = json.load(fp)
    KL, dL, KR, dR = (np.array(s[k]) for k in ("KL", "dL", "KR", "dR"))
    R1, R2, P1, P2 = (np.array(r[k]) for k in ("R1", "R2", "P1", "P2"))
    size = tuple(r["image_size"])
    maps = (cv2.initUndistortRectifyMap(KL, dL, R1, P1, size, cv2.CV_16SC2),
            cv2.initUndistortRectifyMap(KR, dR, R2, P2, size, cv2.CV_16SC2))   # 16SC2: 빠른 remap 용
    fB = P1[0, 0] * r["baseline_mm"]
    return maps, fB, float(P1[0, 0])


def rectify(maps, fL, fR):
    (mLx, mLy), (mRx, mRy) = maps
    return (cv2.remap(fL, mLx, mLy, cv2.INTER_LINEAR), cv2.remap(fR, mRx, mRy, cv2.INTER_LINEAR))


# ---------------- 조밀 시차 (EXP-13 과 같은 설정) ----------------
def make_sgbm(min_disp, num_disp, scale, bs=7, uniq=5):
    md = int(round(min_disp * scale))
    nd = int(np.ceil(num_disp * scale / 16) * 16)
    sg = cv2.StereoSGBM_create(minDisparity=md, numDisparities=nd, blockSize=bs,
                               P1=8 * bs * bs, P2=32 * bs * bs, uniquenessRatio=uniq,
                               speckleWindowSize=100, speckleRange=2, disp12MaxDiff=2,
                               mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    return sg, md


def disparity(sg, md, scale, gL, gR):
    """정렬된 회색 영상 → 원래 해상도 기준 시차(px), 못 찾은 곳 NaN"""
    sL = cv2.resize(gL, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sR = cv2.resize(gR, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    d = sg.compute(sL, sR).astype(np.float32) / 16.0
    d[d < md] = np.nan
    d = cv2.resize(d, (gL.shape[1], gL.shape[0]), interpolation=cv2.INTER_NEAREST) / scale
    return d


def box_distance(disp, box, fB, inner=0.5):
    """박스 가운데 inner 비율 영역의 시차 중앙값 → (Z mm, 채움 %, 사분위 폭 px)"""
    x1, y1, x2, y2 = box
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * inner, (y2 - y1) * inner
    xa, xb = int(max(cx - w / 2, 0)), int(min(cx + w / 2, disp.shape[1]))
    ya, yb = int(max(cy - h / 2, 0)), int(min(cy + h / 2, disp.shape[0]))
    roi = disp[ya:yb, xa:xb]
    ok = roi[np.isfinite(roi)]
    fill = ok.size / max(roi.size, 1) * 100
    if ok.size < 20:
        return np.nan, fill, np.nan, (xa, ya, xb, yb)
    d = np.median(ok)
    iqr = np.percentile(ok, 75) - np.percentile(ok, 25)
    return fB / d, fill, iqr, (xa, ya, xb, yb)


def box_patch(gL, gR, box, fB, min_disp, max_disp, shrink=0.1):
    """박스 통째 매칭: 왼쪽 박스 영역(가장자리 10% 제외)을 오른쪽 영상의 같은 가로줄에서 찾아 시차 하나를 구한다.
    - 조밀 시차는 작은 창(약 2.5cm)마다 짝을 찾아서, 무늬 없는 물체(흰 컵 등)는 창 안에 단서가 없어 실패한다
    - 박스 전체(컵 윤곽·손잡이·그림자 포함)를 한 번에 비교하면 단서가 훨씬 많아서 무늬가 적어도 찾을 수 있다
    - 대신 박스 안 배경까지 같이 비교하므로, 물체가 박스를 대부분 채울 때 정확하다
    반환: (Z mm, 정규화 상관 점수 0~1)"""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    xa, xb = int(x1 + w * shrink), int(x2 - w * shrink)
    ya, yb = int(y1 + h * shrink), int(y2 - h * shrink)
    if xb - xa < 8 or yb - ya < 8:
        return np.nan, 0.0
    tpl = gL[ya:yb, xa:xb]
    sx = max(xa - max_disp, 0)                      # 오른쪽 영상에서 찾을 범위 (같은 가로줄)
    ex = xb - min_disp
    if ex - sx <= tpl.shape[1]:
        return np.nan, 0.0
    res = cv2.matchTemplate(gR[ya:yb, sx:ex], tpl, cv2.TM_CCOEFF_NORMED)[0]
    k = int(np.argmax(res)); score = float(res[k])
    off = 0.0
    if 0 < k < len(res) - 1:                         # 꼭짓점 주변 세 점으로 포물선 → 소수점 위치
        a, b, c = res[k - 1], res[k], res[k + 1]
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if abs(den) > 1e-9 else 0.0
    d = xa - (sx + k + off)
    return (fB / d if d > 0 else np.nan), score


def detect(model, img, conf, cls_filter, agnostic=False):
    """YOLO → [(이름, 확신도, (x1,y1,x2,y2)), ...]"""
    res = model(img, conf=conf, agnostic_nms=agnostic, verbose=False)[0]
    out = []
    for b in res.boxes:
        name = res.names[int(b.cls)]
        if cls_filter and name not in cls_filter:
            continue
        out.append((name, float(b.conf), tuple(float(v) for v in b.xyxy[0])))
    return out


def classify(clf, img, found, imgsz=128):
    """2단계: 박스마다 잘라서(EXP-17과 같은 crop_sign) 분류기로 종류를 다시 정한다"""
    if clf is None or not found:
        return found
    from make_cls_dataset import crop_sign
    res = clf.predict([crop_sign(img, b) for _, _, b in found], imgsz=imgsz, verbose=False)
    return [(clf.names[int(r.probs.top1)], cf, b) for (_, cf, b), r in zip(found, res)]


def measure(a, disp, gL, gR, found, fB, fx):
    dets = []
    for name, cf, box in found:
        Z, fill, iqr, inner = box_distance(disp, box, fB)
        Zp, sc = box_patch(gL, gR, box, fB, a.min_disp, a.min_disp + a.num_disp)
        Zm = fx * a.size_mm / (box[2] - box[0]) if a.size_mm else np.nan   # 단안: Z = fx·H/w
        dets.append((name, cf, box, Z, fill, iqr, inner, Zp, sc, Zm))
    return dets


def draw(img, dets, invalid_w):
    vis = img.copy()
    cv2.rectangle(vis, (0, 0), (invalid_w, vis.shape[0]), (60, 60, 60), 2)   # 시차 계산 불가 띠 표시
    cv2.putText(vis, "no-stereo zone", (10, vis.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 120), 2)
    for name, cf, box, Z, fill, _, inner, Zp, sc, Zm in dets:
        x1, y1, x2, y2 = map(int, box)
        dense_ok = np.isfinite(Z) and fill >= 50
        patch_ok = np.isfinite(Zp) and sc >= 0.6
        col = (0, 220, 0) if dense_ok else (0, 200, 255) if patch_ok else (0, 0, 255)   # 초록/주황/빨강
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 3)
        cv2.rectangle(vis, inner[:2], inner[2:], col, 1)
        t1 = f"{name} D {Z/1000:.2f}m ({fill:.0f}%)" if np.isfinite(Z) else f"{name} D ? ({fill:.0f}%)"
        t2 = f"P {Zp/1000:.2f}m ({sc:.2f})" if np.isfinite(Zp) else "P ?"
        if np.isfinite(Zm):
            t2 += f" | M {Zm/1000:.2f}m"
        cv2.putText(vis, t1, (x1, max(y1 - 45, 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 3)
        cv2.putText(vis, t2, (x1, max(y1 - 10, 65)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 3)
    return vis


# ---------------- 실시간 모드 ----------------
def run_live(a, model, clf, maps, fB, fx, sg, md):
    def open_cam(i):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 15)
        if not cap.isOpened():
            raise SystemExit(f"[오류] /dev/video{i} 를 열 수 없습니다.")
        return cap
    capL, capR = open_cam(a.left), open_cam(a.right)
    out = os.path.join(ROOT, "captures", a.name)
    os.makedirs(os.path.join(out, "L"), exist_ok=True); os.makedirs(os.path.join(out, "R"), exist_ok=True)
    idx = len(glob.glob(os.path.join(out, "L", "img_*.png")))
    print(f"저장 폴더: {out}  (SPACE 원본 저장 / d 시차 지도 보기 켜기·끄기 / q 종료)")
    t0, n, show_disp = time.time(), 0, False
    while True:
        if not (capL.grab() and capR.grab()):
            continue
        okL, fL = capL.retrieve(); okR, fR = capR.retrieve()
        if not (okL and okR):
            continue
        rL, rR = rectify(maps, fL, fR)
        gL, gR = cv2.cvtColor(rL, cv2.COLOR_BGR2GRAY), cv2.cvtColor(rR, cv2.COLOR_BGR2GRAY)
        disp = disparity(sg, md, a.scale, gL, gR)
        found = classify(clf, rL, detect(model, rL, a.conf, None, a.agnostic))
        if a.cls:
            found = [f for f in found if f[0] in a.cls]
        dets = measure(a, disp, gL, gR, found, fB, fx)
        vis = draw(rL, dets, a.min_disp + a.num_disp)
        n += 1
        fps = n / (time.time() - t0)
        cv2.putText(vis, f"{fps:.1f} fps | saved {idx}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 0), 3)
        if show_disp:   # 진단용: 오른쪽에 시차 지도 (검정 = 시차 없음, 빨강 = 가까움, 파랑 = 멂)
            v = np.nan_to_num((disp - a.min_disp) / a.num_disp * 255, nan=0).clip(0, 255).astype(np.uint8)
            col = cv2.applyColorMap(v, cv2.COLORMAP_JET); col[~np.isfinite(disp)] = 0
            for d_ in dets:
                cv2.rectangle(col, tuple(map(int, d_[2][:2])), tuple(map(int, d_[2][2:])), (255, 255, 255), 3)
            vis = np.hstack([vis, col])
        cv2.imshow("stereo_yolo", cv2.resize(vis, None, fx=a.view_scale * (0.6 if show_disp else 1), fy=a.view_scale * (0.6 if show_disp else 1)))
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("d"):
            show_disp = not show_disp
        if k == ord(" "):
            cv2.imwrite(os.path.join(out, "L", f"img_{idx:03d}.png"), fL)
            cv2.imwrite(os.path.join(out, "R", f"img_{idx:03d}.png"), fR)
            print(f"저장 img_{idx:03d}: " + ", ".join(f"{d[0]} 조밀 {d[3]/10:.1f}cm({d[4]:.0f}%) / 박스 {d[7]/10:.1f}cm({d[8]:.2f})"
                                                     + (f" / 단안 {d[9]/10:.1f}cm" if np.isfinite(d[9]) else "") for d in dets))
            idx += 1
    capL.release(); capR.release(); cv2.destroyAllWindows()


# ---------------- 폴더 분석 모드 ----------------
def run_folder(a, model, clf, maps, fB, fx, sg, md):
    files = sorted(glob.glob(os.path.join(ROOT, "captures", a.folder, "L", "img_*.png")))
    if not files:
        raise SystemExit(f"[오류] captures/{a.folder}/L 에 사진이 없습니다.")
    if a.tape and len(a.tape) != len(files):
        raise SystemExit(f"[오류] 사진 {len(files)}쌍 ≠ --tape {len(a.tape)}개")
    os.makedirs(os.path.join(ROOT, "figures", a.folder), exist_ok=True)
    rows = []
    print(f"f·B = {fB:,.0f} / 시차 {a.min_disp}~{a.min_disp + a.num_disp}px, 축소 {a.scale}\n")
    print(f"{'사진':<12}{'물체':<10}{'확신':>5}{'조밀Z':>9}{'채움':>6}{'박스Z':>9}{'점수':>6}" + (f"{'단안Z':>9}" if a.size_mm else "")
          + ("   줄자   조밀 오차   박스 오차" + ("   단안 오차" if a.size_mm else "") if a.tape else ""))
    for i, fLp in enumerate(files):
        fL = cv2.imread(fLp); fR = cv2.imread(fLp.replace(f"{os.sep}L{os.sep}", f"{os.sep}R{os.sep}"))
        rL, rR = rectify(maps, fL, fR)
        gL, gR = cv2.cvtColor(rL, cv2.COLOR_BGR2GRAY), cv2.cvtColor(rR, cv2.COLOR_BGR2GRAY)
        disp = disparity(sg, md, a.scale, gL, gR)
        found = [("box", 1.0, tuple(a.box))] if a.box else classify(clf, rL, detect(model, rL, a.conf, None, a.agnostic))
        if a.cls and not a.box:
            found = [f for f in found if f[0] in a.cls]
        dets = measure(a, disp, gL, gR, found, fB, fx)
        if not dets:
            print(f"{os.path.basename(fLp):<12}(검출 없음)")
        for name, cf, box, Z, fill, iqr, _, Zp, sc, Zm in dets:
            line = f"{os.path.basename(fLp):<12}{name:<10}{cf:5.2f}{Z/10:7.1f}cm{fill:5.0f}%{Zp/10:7.1f}cm{sc:6.2f}"
            if a.size_mm:
                line += f"{Zm/10:7.1f}cm"
            row = {"file": os.path.basename(fLp), "cls": name, "conf": cf, "Z_mm": float(Z), "fill": fill,
                   "iqr_px": float(iqr), "Zp_mm": float(Zp), "patch_score": sc, "Zm_mm": float(Zm), "box": box}
            if a.tape:
                t = a.tape[i] * 10
                line += f"{a.tape[i]:6.0f}cm{(Z - t):+9.1f}mm{(Zp - t):+10.1f}mm"
                if a.size_mm:
                    line += f"{(Zm - t):+10.1f}mm"
                row["tape_mm"] = t
            print(line)
            rows.append(row)
        cv2.imwrite(os.path.join(ROOT, "figures", a.folder, os.path.basename(fLp)),
                    cv2.resize(draw(rL, dets, a.min_disp + a.num_disp), None, fx=0.5, fy=0.5))
    if a.tape:
        # 사진마다 확신도가 가장 높은 검출 하나로 요약
        best = {}
        for r in rows:
            if r["file"] not in best or r["conf"] > best[r["file"]]["conf"]:
                best[r["file"]] = r
        print()
        for key, label in (("Z_mm", "조밀"), ("Zp_mm", "박스 매칭")) + ((("Zm_mm", "단안(크기)"),) if a.size_mm else ()):
            g = [b for b in best.values() if np.isfinite(b[key])]
            if len(g) < 2:
                print(f"요약 [{label}]: 성공 {len(g)}/{len(best)}장 → 요약 불가")
                continue
            T = np.array([b["tape_mm"] for b in g]); Zs = np.array([b[key] for b in g])
            e = Zs - T
            A = np.vstack([T, np.ones_like(T)]).T
            (s_, b_), *_ = np.linalg.lstsq(A, Zs, rcond=None)
            print(f"요약 [{label}] 성공 {len(g)}/{len(best)}장: 평균 오차 {e.mean():+.1f}mm, "
                  f"최대 |오차| {np.abs(e).max():.1f}mm ({np.abs(e / T).max() * 100:.1f}%), "
                  f"배율 s = {s_:.4f} ({(s_ - 1) * 100:+.2f}%), 일정 오차 b = {b_:+.1f}mm")
    with open(os.path.join(ROOT, "results", f"{a.folder}_yolo.json"), "w", encoding="utf-8") as fp:
        json.dump(rows, fp, ensure_ascii=False, indent=1, default=float)
    print(f"\n그림: figures/{a.folder}/   결과: results/{a.folder}_yolo.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stereo", required=True, help="예: EXP-10_stereo")
    ap.add_argument("--left", type=int, default=0); ap.add_argument("--right", type=int, default=6)
    ap.add_argument("--name", default="EXP-15_live", help="실시간 모드 저장 폴더")
    ap.add_argument("--folder", default=None, help="폴더 분석 모드: captures/<folder>")
    ap.add_argument("--tape", nargs="+", type=float, default=None, help="사진 순서대로 줄자 거리 cm")
    ap.add_argument("--box", nargs=4, type=float, default=None, help="YOLO 대신 직접 박스 지정 x1 y1 x2 y2 (정렬 영상 좌표)")
    ap.add_argument("--cls", nargs="+", default=None, help="이 종류만 표시 (예: bottle cup book)")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--cls-model", default=None, help="분류기 best.pt (2단계: 박스를 잘라 종류를 다시 판단)")
    ap.add_argument("--size-mm", type=float, default=None, help="물체 실제 가로 크기 mm (표지판 120) → 단안 거리 같이 계산")
    ap.add_argument("--conf", type=float, default=0.35, help="YOLO 확신도 기준")
    ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument("--min-disp", type=int, default=96)
    ap.add_argument("--num-disp", type=int, default=480)
    ap.add_argument("--view-scale", type=float, default=0.6)
    a = ap.parse_args()

    maps, fB, fx = load_rectify(a.stereo)
    sg, md = make_sgbm(a.min_disp, a.num_disp, a.scale)
    a.agnostic = os.path.basename(a.model) != "yolo11n.pt"   # 직접 학습한 모델이면 한 물체 = 박스 하나
    model = clf = None
    if not a.box:
        from ultralytics import YOLO
        model = YOLO(a.model)
        clf = YOLO(a.cls_model) if a.cls_model else None
    if a.folder:
        run_folder(a, model, clf, maps, fB, fx, sg, md)
    else:
        run_live(a, model, clf, maps, fB, fx, sg, md)


if __name__ == "__main__":
    main()
