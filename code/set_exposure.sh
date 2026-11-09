#!/usr/bin/env bash
# 재부팅·USB 재연결 후 두 웹캠 노출을 EXP-13 최종값으로 다시 고정한다
# (자동 노출이면 좌우 밝기가 달라져서 시차 계산이 흔들림 → EXP-13 결론)
# 사용: bash code/set_exposure.sh        (기본 카메라 0, 6)
#       bash code/set_exposure.sh 0 4    (번호가 바뀌었을 때)
L=${1:-0}; R=${2:-6}
for d in $L $R; do
  v4l2-ctl -d /dev/video$d -c auto_exposure=1 -c exposure_time_absolute=1200 -c gain=4 \
    && echo "/dev/video$d: 노출 1200, 게인 4 고정" \
    || echo "/dev/video$d: 설정 실패 (번호 확인: v4l2-ctl --list-devices)"
done
v4l2-ctl -d /dev/video$L -C auto_exposure,exposure_time_absolute,gain
