#!/usr/bin/env python3
"""External CPU use (PLAN.md, Amendment 1): host-wide busy cores minus the cores used by processes in this container.
  extload.py sample OUTFILE INTERVAL   append "unix_time load1 host_cores own_cores external_cores" every INTERVAL s until killed
  extload.py check SECONDS             print the median and max external cores over SECONDS (and the 1-minute load average)"""
import os, sys, time, statistics as st
HZ = os.sysconf("SC_CLK_TCK")
def host():
    v = list(map(int, open("/proc/stat").readline().split()[1:9]))
    return sum(v) - v[3] - v[4]                      # everything except idle and iowait (steal included)
def own():
    t = 0
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                x = open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()
                t += int(x[11]) + int(x[12])         # utime + stime
            except Exception:
                pass
    return t
def sample(interval):
    h0, o0, t0 = host(), own(), time.time(); time.sleep(interval); h1, o1, t1 = host(), own(), time.time()
    dt = t1 - t0; hc = (h1 - h0) / (HZ * dt); oc = max(0.0, (o1 - o0) / (HZ * dt))
    return hc, oc, max(0.0, hc - oc)
load1 = lambda: float(open("/proc/loadavg").read().split()[0])
if sys.argv[1] == "sample":
    out, interval = sys.argv[2], float(sys.argv[3])
    while True:
        hc, oc, ec = sample(interval)
        with open(out, "a") as f: f.write(f"{time.time():.0f} {load1():.2f} {hc:.2f} {oc:.2f} {ec:.2f}\n")
else:
    secs = float(sys.argv[2]); xs = [sample(2.0)[2] for _ in range(max(1, int(secs // 2)))]
    print(f"{st.median(xs):.2f} {max(xs):.2f} {load1():.2f}")
