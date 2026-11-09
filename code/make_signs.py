#!/usr/bin/env python3
"""
EXP-16: 속도제한 표지판(30, 50) 인쇄용 A4 PDF 만들기
- 실제 규제표지 모양: 흰 바탕 원 + 빨간 테두리 + 검은 숫자
- 지름 12cm (70~150cm 거리에서 화면에 적당한 크기)
- 페이지마다 30 하나, 50 하나 → 2페이지 = 종류별 2장
- 10cm 눈금자: 인쇄 후 자로 재서 "실제 크기로 인쇄됐는지" 확인 (체커보드 칸 크기 확인과 같은 이유)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, Rectangle

W, H = 210, 297            # A4 (mm)
D = 120                    # 표지판 지름 (mm)
RED, BLACK = "#d71920", "#111111"

def sign(ax, cx, cy, num):
    r = D / 2
    ax.add_patch(Circle((cx, cy), r, color=RED))                    # 빨간 테두리
    ax.add_patch(Circle((cx, cy), r * 0.78, color="white"))         # 흰 안쪽 (테두리 두께 약 지름의 11%)
    ax.add_patch(Circle((cx, cy), r + 1.5, fill=False, ec="#bbbbbb", lw=0.4, ls="--"))  # 자르는 선
    ax.text(cx, cy - 2, str(num), ha="center", va="center", color=BLACK,
            fontsize=150, fontweight="bold", family="DejaVu Sans")

with PdfPages("figures/speed_signs_A4.pdf") as pdf:
    for page in range(2):
        fig = plt.figure(figsize=(W / 25.4, H / 25.4))
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
        sign(ax, W / 2, H - 20 - D / 2, 30)
        sign(ax, W / 2, H - 20 - D - 18 - D / 2, 50)
        # 10cm 눈금자 (인쇄 배율 확인용)
        x0, y0 = 20, 12
        ax.add_patch(Rectangle((x0, y0), 100, 2.5, color=BLACK))
        for i in range(11):
            ax.plot([x0 + 10 * i] * 2, [y0, y0 + (6 if i % 5 == 0 else 4)], color=BLACK, lw=0.6)
        ax.text(x0 + 105, y0 + 1, "100 mm  (print at 100% / actual size)", va="center", fontsize=8)
        pdf.savefig(fig); plt.close(fig)
print("saved figures/speed_signs_A4.pdf")
