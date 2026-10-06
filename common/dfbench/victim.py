"""The single, shared victim model for the whole benchmark.

Standalone re-implementation of MMAction2's Recognizer3D(C3D + I3DHead) that
loads `c3d_sports1m_16x1x1_45e_ucf101_rgb_20201021-26655025.pth` verbatim.
It is written without mmcv so that all six attack environments -- which have
mutually incompatible dependency trees -- can load the *identical* model with
the *identical* weights and preprocessing.  Equivalence with the original
mmaction implementation was verified against the original Recognizer3D
(max |logit difference| = 0.000e+00) in the parent benchmark.

Architecture (Geo-TRAP mmaction/models/backbones/c3d.py, norm_cfg=None so
ConvModule is just Conv3d+ReLU, hence the `.conv.` in the state-dict keys):

  conv1a 3->64   pool1 (1,2,2)
  conv2a 64->128 pool2 (2,2,2)
  conv3a/3b 128->256->256  pool3 (2,2,2)
  conv4a/4b 256->512->512  pool4 (2,2,2)
  conv5a/5b 512->512->512  pool5 (2,2,2) pad (0,1,1)
  flatten -> fc6 8192->4096 -> ReLU -> Dropout -> fc7 4096->4096 -> ReLU
  I3DHead(spatial_type=None): Dropout -> fc_cls 4096->101
"""
import torch
import torch.nn as nn

from .data import MEAN, STD


class C3DBackbone(nn.Module):
    def __init__(self, dropout_ratio=0.5):
        super().__init__()
        k = dict(kernel_size=3, padding=1)

        def conv(i, o):
            m = nn.Module()
            m.conv = nn.Conv3d(i, o, **k)
            return m

        self.conv1a = conv(3, 64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv2a = conv(64, 128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv3a = conv(128, 256)
        self.conv3b = conv(256, 256)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv4a = conv(256, 512)
        self.conv4b = conv(512, 512)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv5a = conv(512, 512)
        self.conv5b = conv(512, 512)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2),
                                  padding=(0, 1, 1))
        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_ratio)

    def forward_features(self, x):
        """Post-ReLU activations of every conv block.

        U3D's objective is a power-normalised feature distance over exactly
        these eight tensors (see u3d/python/video_classification/network/
        C3D_model.py: conv1, conv2, conv3a, conv3b, conv4a, conv4b, conv5a,
        conv5b), so they are exposed here under the same names.
        """
        f = {}
        x = self.relu(self.conv1a.conv(x)); f["conv1"] = x
        x = self.pool1(x)
        x = self.relu(self.conv2a.conv(x)); f["conv2"] = x
        x = self.pool2(x)
        x = self.relu(self.conv3a.conv(x)); f["conv3a"] = x
        x = self.relu(self.conv3b.conv(x)); f["conv3b"] = x
        x = self.pool3(x)
        x = self.relu(self.conv4a.conv(x)); f["conv4a"] = x
        x = self.relu(self.conv4b.conv(x)); f["conv4b"] = x
        x = self.pool4(x)
        x = self.relu(self.conv5a.conv(x)); f["conv5a"] = x
        x = self.relu(self.conv5b.conv(x)); f["conv5b"] = x
        x = self.pool5(x)
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        return x, f

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1a.conv(x)))
        x = self.pool2(self.relu(self.conv2a.conv(x)))
        x = self.relu(self.conv3a.conv(x))
        x = self.pool3(self.relu(self.conv3b.conv(x)))
        x = self.relu(self.conv4a.conv(x))
        x = self.pool4(self.relu(self.conv4b.conv(x)))
        x = self.relu(self.conv5a.conv(x))
        x = self.pool5(self.relu(self.conv5b.conv(x)))
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        return x


class I3DHead(nn.Module):
    """spatial_type=None -> no avg pool, just dropout + fc."""
    def __init__(self, num_classes=101, in_channels=4096, dropout_ratio=0.5):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_ratio) if dropout_ratio else None
        self.fc_cls = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        return self.fc_cls(x.view(x.shape[0], -1))


class Recognizer3D(nn.Module):
    def __init__(self, num_classes=101):
        super().__init__()
        self.backbone = C3DBackbone()
        self.cls_head = I3DHead(num_classes=num_classes)

    def forward(self, x):
        return self.cls_head(self.backbone(x))


class QueryBudgetExceeded(BaseException):
    """Raised when an attack exceeds the query budget it was given.

    Deliberately derived from BaseException, not Exception: several of the
    attack repos wrap their optimisation loop in a bare ``except Exception``,
    which would swallow the budget signal and spin forever.
    """


class C3DVictim:
    """Query interface every attack in this benchmark must go through.

    `query(clips)` takes RGB pixels in [0, 255], shape (N,3,16,112,112), and
    returns raw logits (N,101).  Every clip in the batch counts as one query --
    batching is a speed optimisation, never a discount.
    """

    def __init__(self, ckpt_path=None, device="cuda", max_batch=64,
                 query_budget=None, clamp=True):
        from .paths import C3D_CKPT
        self.device = torch.device(device)
        self.model = Recognizer3D().to(self.device).eval()
        ck = torch.load(ckpt_path or C3D_CKPT, map_location="cpu", weights_only=False)
        sd = ck["state_dict"] if "state_dict" in ck else ck
        missing, unexpected = self.model.load_state_dict(sd, strict=True), None
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.max_batch = max_batch
        self.query_budget = query_budget
        self.clamp = clamp
        self._mean = torch.tensor(MEAN, device=self.device).view(1, 3, 1, 1, 1)
        self._std = torch.tensor(STD, device=self.device).view(1, 3, 1, 1, 1)
        self.reset()

    # ------------------------------------------------------------- accounting
    def reset(self):
        self.n_queries = 0
        self.first_success_query = None
        self.first_success_clip = None    # so MAP is available even if the
                                          # attack returns nothing at the end
        # eps 검사를 뺀 '처음 오분류된 시점'. FR_0 기준의 질의예산 곡선을
        # 그리려면 이 값이 필요하다. 성공 판정(first_success_query)과 독립.
        self.first_misclass_query = None
        self.first_misclass_clip = None
        self._target_label = None
        self._success_test = None

    def arm(self, true_label, success_test=None):
        """Start tracking success for one video.

        success_test(logits_row, clip) -> bool decides whether a query counts as
        a valid adversarial example.  Default: any misclassification.
        """
        self._target_label = int(true_label)
        self._success_test = success_test
        self.first_success_query = None
        self.first_misclass_query = None

    # ------------------------------------------------------------------ query
    @torch.no_grad()
    def query(self, clips, count=True):
        """clips: (N,3,16,112,112) float, RGB pixel space [0,255]."""
        x = self._to_tensor(clips)
        n = x.shape[0]
        if count and self.query_budget is not None and self.n_queries + n > self.query_budget:
            raise QueryBudgetExceeded(
                f"{self.n_queries}+{n} > budget {self.query_budget}")
        outs = []
        for i in range(0, n, self.max_batch):
            chunk = x[i:i + self.max_batch]
            outs.append(self.model((chunk - self._mean) / self._std))
        logits = torch.cat(outs, 0)
        if count:
            self._account(logits, x)
        return logits

    def _account(self, logits, x):
        n = logits.shape[0]
        base = self.n_queries
        self.n_queries += n
        if self._target_label is None:
            return
        preds = logits.argmax(1)
        if self.first_misclass_query is None:          # eps 무관, 오분류만
            for j in range(n):
                if int(preds[j]) != self._target_label:
                    self.first_misclass_query = base + j + 1
                    self.first_misclass_clip = x[j].detach().clone()
                    break
        if self.first_success_query is not None:
            return
        for j in range(n):
            if self._success_test is not None:
                ok = self._success_test(logits[j], x[j])
            else:
                ok = int(preds[j]) != self._target_label
            if ok:
                self.first_success_query = base + j + 1
                self.first_success_clip = x[j].detach().clone()
                return

    @torch.no_grad()
    def note_iterate(self, adv):
        """Tell the victim about the attack's CURRENT candidate solution.

        This performs an *uncounted* forward pass and, if the candidate is a
        valid adversarial example, records the query count at which the attack
        first held one.  It exists because gradient-estimation attacks (NES and
        friends) probe *away* from their iterate: no individual queried point
        need satisfy the epsilon constraint even while the iterate itself
        already does.  QS asks "how many queries did the attack need in order to
        succeed", not "which single query was itself successful", so success has
        to be judged on the iterate.

        Adds no queries to the attack's budget, and adapters may call it as
        often as they like.
        """
        if self._target_label is None:
            return False
        x = self._to_tensor(adv)
        logits = self.model((x - self._mean) / self._std)
        if self.first_misclass_query is None:
            for j in range(x.shape[0]):
                if int(logits[j].argmax()) != self._target_label:
                    self.first_misclass_query = max(self.n_queries, 1)
                    self.first_misclass_clip = x[j].detach().clone()
                    break
        if self.first_success_query is not None:
            return False
        for j in range(x.shape[0]):
            ok = (self._success_test(logits[j], x[j]) if self._success_test is not None
                  else int(logits[j].argmax()) != self._target_label)
            if ok:
                self.first_success_query = max(self.n_queries, 1)
                self.first_success_clip = x[j].detach().clone()
                return True
        return False

    def _to_tensor(self, clips):
        if not torch.is_tensor(clips):
            clips = torch.as_tensor(clips)
        x = clips.to(self.device, dtype=torch.float32)
        if x.dim() == 4:
            x = x.unsqueeze(0)
        if self.clamp:
            x = x.clamp(0.0, 255.0)
        return x

    # --------------------------------------------------------------- helpers
    @torch.no_grad()
    def features(self, clips):
        """(logits, {layer: activation}) for the given pixel-space clips.

        Uncounted: this is white-box model access, which the U3D reference
        implementation's PSO objective requires.  Reported separately.
        """
        x = self._to_tensor(clips)
        feat, d = self.model.backbone.forward_features((x - self._mean) / self._std)
        return self.model.cls_head(feat), d

    @torch.no_grad()
    def predict(self, clips):
        """Uncounted forward pass -- for measuring clean accuracy only."""
        return self.query(clips, count=False)

    @torch.no_grad()
    def predict_video(self, clips, average="score"):
        """Average logits/scores over multiple clips of one video (uncounted)."""
        logits = self.predict(clips)
        if average == "prob":
            return torch.softmax(logits, 1).mean(0)
        return logits.mean(0)
