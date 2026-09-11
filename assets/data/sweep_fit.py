#!/usr/bin/env python3
"""
sweep_fit.py — 抓取舵机扫描数据并拟合逼近函数

用法：
    # 1) 抓数据（板子上烧 servo_sweep_test.ino，复位后立刻运行）
    python3 sweep_fit.py capture -o sweep.csv

    # 2) 分析已有的 CSV
    python3 sweep_fit.py fit sweep.csv

    # 3) 抓完直接分析
    python3 sweep_fit.py capture -o sweep.csv --then-fit

依赖：
    pip3 install pyserial numpy
    pip3 install matplotlib      # 可选，画图用
"""

import argparse
import csv
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.exit("缺少 numpy，请运行：pip3 install numpy")


# ══════════════════════════════════════════════════════
#  抓取
# ══════════════════════════════════════════════════════

def pick_port():
    try:
        from serial.tools import list_ports
    except ImportError:
        sys.exit("缺少 pyserial，请运行：pip3 install pyserial")

    ports = [p for p in list_ports.comports()
             if "usbserial" in p.device or "usbmodem" in p.device
             or "wch" in p.device.lower()]
    if not ports:
        ports = list(list_ports.comports())
    if not ports:
        sys.exit("找不到串口设备，检查 USB 线和驱动")
    if len(ports) == 1:
        return ports[0].device

    print("找到多个串口：")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p.device}  {p.description}")
    idx = int(input("选哪个？ ") or 0)
    return ports[idx].device


def capture(out_path, port=None, baud=115200, timeout=600):
    import serial

    port = port or pick_port()
    print(f"连接 {port} @ {baud}")
    print("等待 '### SWEEP BEGIN'（如果板子已经跑完，按一下 RST 重跑）...")

    rows = []
    header = None
    started = False

    with serial.Serial(port, baud, timeout=2) as ser:
        import time
        t0 = time.time()
        while time.time() - t0 < timeout:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue

            if line.startswith("### SWEEP BEGIN"):
                started = True
                rows.clear()
                header = None
                print("开始接收...")
                continue
            if line.startswith("### SWEEP END"):
                print(f"接收完毕，共 {len(rows)} 行")
                break
            if line.startswith("###"):
                print("  " + line[3:].strip())
                continue
            if not started:
                continue

            if header is None and line.startswith("axis,"):
                header = line.split(",")
                continue

            parts = line.split(",")
            if header and len(parts) == len(header):
                rows.append(parts)
                if len(rows) % 20 == 0:
                    print(f"  {len(rows)} 点...", end="\r", flush=True)
        else:
            print("\n超时了，把已收到的数据存下来")

    if not rows:
        sys.exit("没收到数据。确认板子已烧录扫描程序，并在运行后按 RST 重启。")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"\n已保存 {out_path}（{len(rows)} 行）")
    return out_path


# ══════════════════════════════════════════════════════
#  分析
# ══════════════════════════════════════════════════════

def load(path):
    data = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["axis"], r["dir"])
            data.setdefault(key, []).append({
                "deg": float(r["deg"]),
                "adc": float(r["adc_mean"]),
                "sd":  float(r["adc_sd"]),
                "mv":  float(r["mv"]),
            })
    return data


def fit_report(axis, up, dn):
    """对一个轴做拟合和回差分析"""
    print(f"\n{'═' * 58}")
    print(f"  {axis} 轴")
    print(f"{'═' * 58}")

    both = up + dn
    deg = np.array([p["deg"] for p in both])
    adc = np.array([p["adc"] for p in both])

    # ── 噪声 ──
    sds = np.array([p["sd"] for p in both])
    print(f"\n噪声：ADC 标准差 平均 {sds.mean():.1f}, 最大 {sds.max():.1f}")

    # ── 回差：同一角度，顺扫和逆扫的 ADC 差 ──
    up_map = {p["deg"]: p["adc"] for p in up}
    dn_map = {p["deg"]: p["adc"] for p in dn}
    common = sorted(set(up_map) & set(dn_map))
    if common:
        diffs = np.array([up_map[d] - dn_map[d] for d in common])
        # 用整体斜率把 ADC 差换算成角度
        slope = np.polyfit(deg, adc, 1)[0]      # counts per degree
        bl_deg = diffs / slope
        print(f"灵敏度：{slope:.2f} counts/度  →  1 count ≈ {1/slope:.3f}°")
        print(f"回差 backlash：平均 {abs(bl_deg.mean()):.2f}°, "
              f"最大 {abs(bl_deg).max():.2f}°")
        print(f"  （顺扫 ADC 比逆扫平均高 {diffs.mean():+.1f} counts）")
        worst = common[int(np.argmax(np.abs(bl_deg)))]
        print(f"  回差最大处在 {worst:.0f}°")

    # ── 多项式拟合，比较阶数 ──
    print(f"\n拟合 ADC → 角度（用双向平均，消掉回差偏置）：")
    if common:
        x = np.array(common)
        y = np.array([(up_map[d] + dn_map[d]) / 2 for d in common])
    else:
        x, y = deg, adc

    best = None
    for order in (1, 2, 3):
        # 拟合 角度 = f(ADC)，因为实际使用时是从 ADC 反推角度
        c = np.polyfit(y, x, order)
        pred = np.polyval(c, y)
        resid = pred - x
        rms = np.sqrt((resid ** 2).mean())
        mx = np.abs(resid).max()
        print(f"  {order} 阶: RMS 误差 {rms:.3f}°, 最大 {mx:.3f}°")
        if best is None or rms < best[1] * 0.85:   # 明显更好才升阶
            best = (order, rms, c)

    order, rms, coef = best
    print(f"\n→ 推荐 {order} 阶（再升阶收益不明显）")

    # ── 生成 C 代码 ──
    name = axis.lower()
    terms = []
    n = len(coef) - 1
    for i, c in enumerate(coef):
        p = n - i
        if p == 0:
            terms.append(f"{c:.6g}f")
        elif p == 1:
            terms.append(f"{c:.6g}f * r")
        else:
            terms.append(f"{c:.6g}f * r{p}")

    print(f"\n// {axis} 轴：ADC → 角度，{order} 阶多项式，RMS {rms:.3f}°")
    print(f"float {name}RawToDeg(int raw) {{")
    print(f"  float r = (float)raw;")
    if n >= 2:
        print(f"  float r2 = r * r;")
    if n >= 3:
        print(f"  float r3 = r2 * r;")
    print(f"  return {' + '.join(reversed(terms))};")
    print(f"}}")

    return {"axis": axis, "x": x, "y": y, "coef": coef,
            "order": order, "rms": rms,
            "up": up, "dn": dn}


def plot(results, path="sweep_plot.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n（没装 matplotlib，跳过画图：pip3 install matplotlib）")
        return

    fig, axes = plt.subplots(2, len(results), figsize=(6 * len(results), 8))
    if len(results) == 1:
        axes = axes.reshape(2, 1)

    for j, r in enumerate(results):
        ax = axes[0, j]
        ax.plot([p["deg"] for p in r["up"]], [p["adc"] for p in r["up"]],
                ".-", label="sweep up", ms=3)
        ax.plot([p["deg"] for p in r["dn"]], [p["adc"] for p in r["dn"]],
                ".-", label="sweep down", ms=3)
        ax.set_title(f"{r['axis']}  transfer curve")
        ax.set_xlabel("angle (deg)")
        ax.set_ylabel("ADC")
        ax.grid(alpha=.3)
        ax.legend()

        ax = axes[1, j]
        pred = np.polyval(r["coef"], r["y"])
        ax.plot(r["x"], pred - r["x"], ".-", ms=3)
        ax.axhline(0, color="k", lw=.5)
        ax.set_title(f"{r['axis']}  fit residual ({r['order']} order, "
                     f"RMS {r['rms']:.3f}°)")
        ax.set_xlabel("angle (deg)")
        ax.set_ylabel("error (deg)")
        ax.grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\n图已保存：{path}")


def analyse(path):
    data = load(path)
    axes = sorted({k[0] for k in data})
    results = []
    for a in axes:
        up = sorted(data.get((a, "UP"), []), key=lambda p: p["deg"])
        dn = sorted(data.get((a, "DN"), []), key=lambda p: p["deg"])
        if not up and not dn:
            continue
        results.append(fit_report(a, up, dn))
    if results:
        plot(results)
    print()


# ══════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="舵机扫描数据抓取与拟合")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="从串口抓取")
    c.add_argument("-o", "--output", default="sweep.csv")
    c.add_argument("-p", "--port")
    c.add_argument("-b", "--baud", type=int, default=115200)
    c.add_argument("--then-fit", action="store_true", help="抓完直接分析")

    f = sub.add_parser("fit", help="分析已有 CSV")
    f.add_argument("csv")

    a = ap.parse_args()

    if a.cmd == "capture":
        p = capture(a.output, a.port, a.baud)
        if a.then_fit:
            analyse(p)
    else:
        analyse(a.csv)


if __name__ == "__main__":
    main()
