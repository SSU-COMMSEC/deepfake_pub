"""TensorRT 엔진을 피해 모델로 쓰는 victim.

공격의 **생성 지연**을 정밀도별로 재려면, 공격이 질의하는 victim 자체가 해당
정밀도로 돌아야 한다. C3DVictim 은 PyTorch 모델을 쓰므로 FP32 만 가능하다.
여기서는 self.model 만 TensorRT 엔진으로 갈아끼워 나머지 인터페이스
(query / note_iterate / arm / features …) 를 그대로 재사용한다.

ONNX 는 정규화를 내부에 포함하므로(scripts/export_c3d_onnx.py 의 Wrapped),
C3DVictim 이 넘겨주는 정규화된 입력을 원본 픽셀로 되돌려 엔진에 넣는다
(std=1 이라 x*std+mean 은 정확한 역변환이다).
"""
import numpy as np
import torch
import tensorrt as trt

from .victim import C3DVictim


class _TRTRunner:
    """플랜 파일 하나를 감싸 (B,3,16,112,112) 픽셀 -> (B,101) 로짓."""

    def __init__(self, plan_path, device):
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(plan_path, "rb") as f, trt.Runtime(self.logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"엔진 로드 실패: {plan_path}")
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i)
                 for i in range(self.engine.num_io_tensors)]
        self.i_name = [n for n in names
                       if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT][0]
        self.o_names = [n for n in names
                        if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        # 특징 출력 엔진(c3d_feat)은 로짓 + conv1..conv5b 를 함께 내보낸다.
        self.o_name = ("logits" if "logits" in self.o_names else self.o_names[0])
        self.device = device
        self._out = {}          # 배치별 출력 버퍼 재사용

    def __call__(self, x):
        """x: (B,3,16,112,112) float32 CUDA 텐서, [0,255] 픽셀."""
        x = x.contiguous().to(self.device, torch.float32)
        b = x.shape[0]
        self.ctx.set_input_shape(self.i_name, tuple(x.shape))
        oshape = tuple(self.ctx.get_tensor_shape(self.o_name))
        bufs = self._out.get(b)
        shapes = {n: tuple(self.ctx.get_tensor_shape(n)) for n in self.o_names}
        if bufs is None or any(tuple(bufs[n].shape) != shapes[n] for n in self.o_names):
            bufs = {n: torch.empty(shapes[n], dtype=torch.float32, device=self.device)
                    for n in self.o_names}
            self._out[b] = bufs
        self.ctx.set_tensor_address(self.i_name, x.data_ptr())
        for n in self.o_names:
            self.ctx.set_tensor_address(n, bufs[n].data_ptr())
        stream = torch.cuda.current_stream(self.device)
        if not self.ctx.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT 실행 실패")
        stream.synchronize()
        self.last = bufs
        return bufs[self.o_name]


class _DenormWrapper(torch.nn.Module):
    """C3DVictim 이 넘기는 정규화 입력을 픽셀로 되돌려 엔진에 전달한다."""

    def __init__(self, runner, mean, std):
        super().__init__()
        self.runner, self.mean, self.std = runner, mean, std

    def forward(self, x_norm):
        return self.runner(x_norm * self.std + self.mean)


class TRTVictim(C3DVictim):
    """C3DVictim 과 동일한 인터페이스, 추론만 TensorRT 로 수행한다.

    features() 는 엔진이 중간 특징을 함께 내보낼 때만 동작한다
    (scripts/export_c3d_feat_onnx.py 로 만든 c3d_feat.onnx 기반 엔진).
    로짓만 내보내는 기본 엔진에서는 NotImplementedError 를 낸다.
    """

    FEAT = ["conv1", "conv2", "conv3a", "conv3b",
            "conv4a", "conv4b", "conv5a", "conv5b"]

    def __init__(self, plan_path, device="cuda", max_batch=32,
                 query_budget=None, clamp=True):
        # C3DVictim.__init__ 을 부르지 않는다 — PyTorch 체크포인트(300 MB) 를 읽을
        # 이유가 없고, Jetson 에서는 메모리도 아깝다. 필요한 상태만 직접 세운다.
        from .data import MEAN, STD
        self.device = torch.device(device)
        self.max_batch = max_batch
        self.query_budget = query_budget
        self.clamp = clamp
        self._mean = torch.tensor(MEAN, device=self.device).view(1, 3, 1, 1, 1)
        self._std = torch.tensor(STD, device=self.device).view(1, 3, 1, 1, 1)
        self.plan_path = plan_path
        self._runner = _TRTRunner(plan_path, self.device)
        self.model = _DenormWrapper(self._runner, self._mean, self._std)
        self.reset()

    @property
    def has_features(self):
        return all(n in self._runner.o_names for n in self.FEAT)

    @torch.no_grad()
    def features(self, clips):
        """(logits, {layer: activation}) — C3DVictim.features 와 같은 계약.

        U3D 의 PSO 목적함수가 요구하는 8개 층을 엔진 출력에서 그대로 꺼낸다.
        질의로 세지 않는 백박스 접근이라는 점도 PyTorch 쪽과 동일하다.
        """
        if not self.has_features:
            raise NotImplementedError(
                f"이 엔진은 중간 특징을 내보내지 않는다: {self.plan_path}\n"
                "scripts/export_c3d_feat_onnx.py 로 만든 ONNX 로 빌드할 것")
        x = self._to_tensor(clips)
        logits = self._runner(x)
        return logits, {n: self._runner.last[n] for n in self.FEAT}
