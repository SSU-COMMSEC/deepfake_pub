"""Canonical paths for the UCF-101 / C3D adversarial-attack benchmark."""
import os

# 저장소 루트를 기본값으로 쓴다 (<root>/common/dfbench/paths.py 기준 세 단계 위)
_REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DF_ROOT      = os.environ.get("DF_ROOT", _REPO_ROOT)
DATASET_DIR  = os.path.join(DF_ROOT, "dataset")
VIDEO_DIR    = os.path.join(DATASET_DIR, "UCF-101")
SPLIT_DIR    = os.path.join(DATASET_DIR, "ucfTrainTestlist")
TESTLIST     = os.path.join(SPLIT_DIR, "testlist01.txt")
CLASSIND     = os.path.join(SPLIT_DIR, "classInd.txt")
FRAME_DIR    = os.path.join(DATASET_DIR, "ucf101_frames")   # decoded rawframes (mmaction layout)

CKPT_DIR     = os.path.join(DF_ROOT, "checkpoints")
C3D_CKPT     = os.path.join(CKPT_DIR, "c3d_sports1m_16x1x1_45e_ucf101_rgb_20201021-26655025.pth")

REPO_DIR     = os.path.join(DF_ROOT, "repos")
RESULT_DIR   = os.path.join(DF_ROOT, "results")
LOG_DIR      = os.path.join(DF_ROOT, "logs")
STATE_DIR    = os.path.join(DF_ROOT, "state")
REPORT_DIR   = os.path.join(DF_ROOT, "reports")

# The frozen evaluation manifest produced by stage 01 (clean accuracy pass).
CLEAN_RESULTS = os.path.join(RESULT_DIR, "00_victim", "clean_predictions.jsonl")
# 평가 대상 집합. DF_EVAL_SUBSET 으로 다른 파일(예: CLVA 식 무작위 100편)을 가리킬 수 있다.
EVAL_SUBSET   = os.environ.get(
    "DF_EVAL_SUBSET",
    os.path.join(RESULT_DIR, "00_victim", "eval_subset.json"))


def load_class_index():
    """-> (name2idx, idx2name) using UCF-101 classInd.txt, 0-based indices."""
    name2idx, idx2name = {}, {}
    with open(CLASSIND) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            i, name = line.split()
            name2idx[name] = int(i) - 1          # classInd.txt is 1-based
            idx2name[int(i) - 1] = name
    return name2idx, idx2name


def load_testlist():
    """-> list of (relpath, label) for official split-1 test list (3783 entries)."""
    name2idx, _ = load_class_index()
    items = []
    with open(TESTLIST) as f:
        for line in f:
            rel = line.strip().split()[0] if line.strip() else ""
            if not rel:
                continue
            rel = rel.replace("\\", "/")
            cls = rel.split("/")[0]
            items.append((rel, name2idx[cls]))
    return items
