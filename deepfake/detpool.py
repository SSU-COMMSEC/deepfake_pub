"""A pool of SimSwap face detectors (onnxruntime-CPU) for many frames at once."""

import multiprocessing as mp
import os

import cv2
import numpy as np

_aligner = None
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def spawn_pool(workers, **kw):
    """A spawn Pool whose workers start with one BLAS/OpenMP thread each. Otherwise every worker's
    OpenBLAS (numpy's, and dlib's face encoder) opens one thread per core: 12 dlib workers made
    336 spinning threads on 28 cores. The variables only reach the children (this process has
    already loaded its BLAS) and are restored right after the workers start."""
    saved = {k: os.environ.get(k) for k in _THREAD_VARS}
    os.environ.update(dict.fromkeys(_THREAD_VARS, "1"))
    try:
        return mp.get_context("spawn").Pool(workers, **kw)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _init(threads):
    global _aligner
    cv2.setNumThreads(1)
    from deepfake.face import FaceAligner, limit_onnx_threads
    limit_onnx_threads(threads)
    _aligner = FaceAligner()


def _detect(frame):
    r = _aligner.detect_full(frame)
    if r is None:
        return None
    r["M"] = r["M"].tolist()
    return r


class DetPool:
    def __init__(self, workers=12, threads=2):
        self.pool = spawn_pool(workers, initializer=_init, initargs=(threads,))

    def detect(self, frames_bgr: list[np.ndarray]) -> list[dict | None]:
        return self.pool.map(_detect, frames_bgr, chunksize=2)

    def close(self):
        self.pool.close()
        self.pool.join()
