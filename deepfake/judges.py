"""PhantomSeal's two identity judges with per-image embedding caching.

Decisions are identical to PhantomSeal's src/evaluate.py (checked in selftest.py):
  FaceNet-512 : facenet-pytorch MTCNN -> InceptionResnetV1(vggface2), match if L2 <= 1.04;
                no face in either image -> distance inf -> counted, not a match
                (Effectiveness.get_images_distance / _get_facenet_matching)
  Face Recognition (dlib): face_encodings(model="large")[0], match if distance <= 0.6
                (compare_faces default); no face -> counted, not a match (_get_facerec_matching)
PhantomSeal recomputes both embeddings for every pair; here each image is embedded once, and
the dlib encodings (CPU) run in a process pool.
"""

import math

import numpy as np
import torch

from deepfake.detpool import spawn_pool

DLIB_TOL = 0.6


class FaceNetJudge:
    def __init__(self, effectiveness, threshold: float):
        self.mtcnn = effectiveness.mtcnn
        self.net = effectiveness.FaceVerification
        self.thr = threshold

    @torch.no_grad()
    def embed(self, img: torch.Tensor):
        arr = img.detach().float().cpu().numpy().transpose(1, 2, 0) * 255.0
        crop = self.mtcnn(arr)
        if crop is None:
            return None
        return self.net(crop.unsqueeze(0).cuda()).detach().cpu()

    def embed_many(self, imgs):
        return [self.embed(x) for x in imgs]

    @staticmethod
    def distance(e1, e2) -> float:
        if e1 is None or e2 is None:
            return math.inf
        return float((e1 - e2).norm().item())

    def match(self, d: float) -> bool:
        return d <= self.thr


def to_u8(img: torch.Tensor) -> np.ndarray:
    """PhantomSeal's conversion in _get_facerec_matching (truncation, not rounding)."""
    return (img.detach().float().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _dlib_encode(img_u8):
    import face_recognition
    encs = face_recognition.face_encodings(img_u8, model="large")
    return encs[0] if len(encs) else None


def dlib_match(e1, e2) -> bool:
    if e1 is None or e2 is None:
        return False
    return bool(np.linalg.norm(e1 - e2) <= DLIB_TOL)


class DlibPool:
    """face_recognition encodings in worker processes (spawned before CUDA is initialised)."""

    def __init__(self, workers=12):
        self.pool = spawn_pool(workers)

    def submit(self, imgs_u8: list[np.ndarray]):
        return self.pool.map_async(_dlib_encode, imgs_u8, chunksize=1)

    def encode(self, imgs_u8):
        return self.submit(imgs_u8).get()

    def close(self):
        self.pool.close()
        self.pool.join()
