"""Faithful re-implementation of the MMAction2 UCF-101 / C3D test pipeline.

Ported 1:1 from Geo-TRAP's MMAction2 fork (v0.9.0), which is the reference
implementation for the checkpoint this benchmark uses:

  SampleFrames(clip_len=16, frame_interval=1, num_clips=N, test_mode=True)
  RawFrameDecode          -> mmcv.imfrombytes(..., channel_order='rgb')  => RGB
  Resize(scale=(128, 171))  keep_ratio=True   => 320x240 becomes 171x128
  CenterCrop(crop_size=112)
  Normalize(mean=[104,117,128], std=[1,1,1], to_bgr=False)
  FormatShape('NCTHW')

Frames come from decoding the .avi directly (decord) rather than from
pre-extracted jpgs; the sampling indices are computed identically.
"""
import numpy as np
import cv2

MEAN = np.array([104.0, 117.0, 128.0], dtype=np.float32)   # RGB order
STD = np.array([1.0, 1.0, 1.0], dtype=np.float32)
CLIP_LEN = 16
CROP_SIZE = 112
RESIZE_SCALE = (128, 171)


# --------------------------------------------------------------- frame indices
def get_test_clip_offsets(num_frames, clip_len=CLIP_LEN, frame_interval=1, num_clips=1):
    """Exact port of SampleFrames._get_test_clips."""
    ori_clip_len = clip_len * frame_interval
    avg_interval = (num_frames - ori_clip_len + 1) / float(num_clips)
    if num_frames > ori_clip_len - 1:
        base_offsets = np.arange(num_clips) * avg_interval
        clip_offsets = (base_offsets + avg_interval / 2.0).astype(int)
    else:
        clip_offsets = np.zeros((num_clips,), dtype=int)
    return clip_offsets


def sample_frame_indices(num_frames, clip_len=CLIP_LEN, frame_interval=1, num_clips=1):
    """Exact port of SampleFrames.__call__ (test_mode, out_of_bound_opt='loop').

    Returns 0-based indices of shape (num_clips*clip_len,).
    """
    clip_offsets = get_test_clip_offsets(num_frames, clip_len, frame_interval, num_clips)
    frame_inds = clip_offsets[:, None] + np.arange(clip_len)[None, :] * frame_interval
    frame_inds = np.concatenate(frame_inds)
    frame_inds = frame_inds.reshape((-1, clip_len))
    frame_inds = np.mod(frame_inds, num_frames)           # 'loop'
    frame_inds = np.concatenate(frame_inds)
    return frame_inds.astype(int)                          # start_index 0 for decord


# ------------------------------------------------------------------- geometry
def rescale_size(w, h, scale=RESIZE_SCALE):
    """Exact port of mmcv.rescale_size for a tuple scale."""
    max_long_edge, max_short_edge = max(scale), min(scale)
    factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))
    return int(w * factor + 0.5), int(h * factor + 0.5)


def resize_and_crop(frames):
    """frames: (T,H,W,3) uint8 RGB -> (T,112,112,3) uint8 RGB."""
    t, h, w, _ = frames.shape
    new_w, new_h = rescale_size(w, h)
    out = np.empty((t, new_h, new_w, 3), dtype=frames.dtype)
    for i in range(t):
        out[i] = cv2.resize(frames[i], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    left = (new_w - CROP_SIZE) // 2
    top = (new_h - CROP_SIZE) // 2
    return out[:, top:top + CROP_SIZE, left:left + CROP_SIZE, :]


# ------------------------------------------------------------------ full load
_READER_CACHE = {}


def load_video_clips(path, num_clips=1):
    """Decode a UCF-101 .avi and return the exact clips the victim is fed.

    Returns float32 array (num_clips, 3, 16, 112, 112) of RGB pixel values in
    [0, 255] -- i.e. BEFORE mean subtraction.  This is the canonical space in
    which every attack perturbs and in which MAP is measured.
    """
    import decord
    vr = decord.VideoReader(path, num_threads=1)
    total = len(vr)
    inds = sample_frame_indices(total, num_clips=num_clips)
    frames = vr.get_batch(list(inds)).asnumpy()            # (T,H,W,3) uint8 RGB
    frames = resize_and_crop(frames)                       # (T,112,112,3)
    clips = frames.reshape(num_clips, CLIP_LEN, CROP_SIZE, CROP_SIZE, 3)
    clips = clips.transpose(0, 4, 1, 2, 3)                 # NCTHW
    return np.ascontiguousarray(clips, dtype=np.float32)


def normalize(clips):
    """(N,3,T,H,W) pixel [0,255] RGB -> normalized tensor the C3D expects."""
    m = MEAN.reshape(1, 3, 1, 1, 1)
    s = STD.reshape(1, 3, 1, 1, 1)
    return (clips - m) / s


def denormalize(x):
    m = MEAN.reshape(1, 3, 1, 1, 1)
    s = STD.reshape(1, 3, 1, 1, 1)
    return x * s + m
