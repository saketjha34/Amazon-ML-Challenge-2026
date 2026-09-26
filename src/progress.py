"""
Progress & Timing - tqdm step bars for each pipeline phase plus a per-step timing summary.

Usage:
    prog = PhaseProgress("TRAIN", ["1", "2", "3a", "3", "4", "5", "6"], total_label=6, save_dir=metrics_dir)
    prog.step("1", "Loading Ground Truth labels")
    ...
    prog.step("3a", "Training blocking cap ranker")
    ...
    prog.done()          # closes the bar, prints and saves the timing table

Inner loops use `pbar(...)`, a thin tqdm wrapper with consistent formatting (nested, auto-cleared).
"""

import json
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional

from tqdm.auto import tqdm


def fmt_secs(s: float) -> str:
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m{sec:02d}s" if h else (f"{m}m{sec:02d}s" if m else f"{sec}s")


def pbar(iterable: Optional[Iterable] = None, total: Optional[int] = None, desc: str = "",
         unit: str = "it", leave: bool = False, **kw):
    """Nested progress bar for inner loops (cleared when finished unless leave=True)."""
    return tqdm(iterable, total=total, desc=f"    {desc}", unit=unit, leave=leave, position=1,
                dynamic_ncols=True, file=sys.stdout, mininterval=0.5, **kw)


class PhaseProgress:
    def __init__(self, phase: str, step_keys: List[str], total_label: Optional[int] = None,
                 save_dir: Optional[Path] = None):
        self.phase = phase
        self.keys = list(step_keys)
        self.total_label = total_label or len(step_keys)
        self.save_dir = Path(save_dir) if save_dir else None
        self.t_phase = time.time()
        self.cur_key: Optional[str] = None
        self.cur_desc = ""
        self.t_step = None
        self.times: List[dict] = []
        self.bar = tqdm(total=len(self.keys), desc=f"[{phase}]", unit="step", position=0, leave=True,
                        dynamic_ncols=True, file=sys.stdout,
                        bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} steps [{elapsed} total{postfix}]")

    def _close_current(self):
        if self.cur_key is None:
            return
        dt = time.time() - self.t_step
        self.times.append({"step": self.cur_key, "desc": self.cur_desc, "seconds": round(dt, 2)})
        self.bar.update(1)
        tqdm.write(f"  [done] [{self.cur_key}/{self.total_label}] {self.cur_desc} - {fmt_secs(dt)} "
                   f"(phase elapsed {fmt_secs(time.time() - self.t_phase)})", file=sys.stdout)
        self.cur_key = None

    def step(self, key: str, desc: str):
        """Finish the running step (if any) and start `key`."""
        self._close_current()
        self.cur_key, self.cur_desc, self.t_step = key, desc, time.time()
        self.bar.set_description_str(f"[{self.phase}] [{key}/{self.total_label}] {desc[:60]}")
        tqdm.write(f"\n[{key}/{self.total_label}] {desc}", file=sys.stdout)

    def skip(self, key: str, desc: str = "skipped (cached)"):
        """Count a step as done without timing it (e.g. satisfied by a checkpoint)."""
        self._close_current()
        self.times.append({"step": key, "desc": desc, "seconds": 0.0})
        self.bar.update(1)

    def done(self) -> List[dict]:
        self._close_current()
        total = time.time() - self.t_phase
        self.bar.set_description_str(f"[{self.phase}] done")
        self.bar.close()
        lines = [f"\n  [TIME] {self.phase} timing summary"]
        for t in self.times:
            share = t["seconds"] / total * 100 if total > 0 else 0
            lines.append(f"     [{t['step']:>3}/{self.total_label}] {t['desc'][:55]:55s} {fmt_secs(t['seconds']):>9s}  {share:5.1f}%")
        lines.append(f"     {'TOTAL':>63s} {fmt_secs(total):>9s}")
        print("\n".join(lines))
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            with open(self.save_dir / f"timings_{self.phase.lower()}.json", "w", encoding="utf-8") as f:
                json.dump({"phase": self.phase, "total_seconds": round(total, 2), "steps": self.times}, f, indent=2)
        return self.times
