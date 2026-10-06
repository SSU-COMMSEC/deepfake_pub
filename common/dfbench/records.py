"""Crash-safe, resumable per-video result store.

Every attack writes ONE json object per attacked video, appended to a .jsonl file
with an fsync. If the process dies (OOM, session drop, power loss) the file is
still a valid prefix, and `done_videos()` tells the runner exactly what to skip
on restart.  Nothing is ever held only in memory.
"""
import json
import os
import tempfile
import time


class RecordStore:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._fh = None

    # ------------------------------------------------------------------ read
    def load(self):
        """-> list of records; silently drops a torn trailing line."""
        recs = []
        if not os.path.exists(self.path):
            return recs
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    # torn final write from a hard kill - ignore, it will be redone
                    continue
        return recs

    def done_videos(self):
        return {r["video"] for r in self.load() if r.get("status") == "done"}

    # ----------------------------------------------------------------- write
    def append(self, rec):
        rec.setdefault("wall_clock", time.strftime("%Y-%m-%d %H:%M:%S"))
        if self._fh is None:
            self._fh = open(self.path, "a")
        self._fh.write(json.dumps(rec, ensure_ascii=False, default=_jsonable) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def _jsonable(o):
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)


def atomic_write_json(path, obj):
    """Write json so a crash mid-write can never leave a corrupt file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, default=_jsonable)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def new_attack_record(video, label, index):
    """The canonical schema every attack must fill in."""
    return {
        "video": video,            # 'ApplyEyeMakeup/v_....avi'
        "index": index,            # position in the frozen eval subset
        "true_label": label,
        "status": "done",          # done | error | skipped
        # --- outcome -----------------------------------------------------
        "success": None,           # bool: misclassified AND within the eps budget
        "final_pred": None,
        # --- query accounting --------------------------------------------
        "queries_used": None,      # total victim queries spent on this video
        "queries_to_success": None,# query index of the FIRST valid adv example
        # --- distortion (all in 0-255 pixel units, over the whole clip) ---
        "map": None,               # ||delta||_1 / numel      <- the MAP metric
        "linf": None,
        "l2": None,
        "n_elements": None,        # T*H*W*C
        # --- bookkeeping --------------------------------------------------
        "wall_seconds": None,
        "error": None,
    }
