"""U3D 논문(Xie et al., IEEE S&P'22) Eq.1 / Eq.2 를 그대로 구현한 벡터화 Perlin 생성기.

`alarst13/u3d` 의 Rust 구현(`rust/perlin/src/perlin3d.rs`)과 세 곳에서 다르다.

  1) :26-32  `p[i] = PERMUTATION[i % 255]`  ->  Ken Perlin 표준은 `% 256`.
     `% 255` 면 doubled table 이 p[i+256] == p[i] 를 하나도 만족하지 않고
     순열 값 하나(180)는 영영 쓰이지 않는다. 해시 일관성이 깨진다.
  2) :152    `(lerp(y1,y2,w) + 1.0) / 2.0`  ->  논문 p(x,y,t) 는 [-1,1] 양극성.
  3) :157    `sin(v * 2*pi / color_period)` ->  논문 Eq.2 는 `sin(p * 2*pi*phi)`.

mode="repo"  : 위 세 가지를 그대로 재현한다 (Rust 출력과 일치 — 검증용).
mode="paper" : 논문 수식대로 고친다.
"""
import numpy as np

_PERMUTATION = np.array([
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,
    142,8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,
    203,117,35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,
    74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,
    220,105,92,41,55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,
    132,187,208,89,18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,
    186,3,64,52,217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,
    59,227,47,16,58,17,182,189,28,42,223,183,170,213,119,248,152,2,44,154,163,
    70,221,153,101,155,167,43,172,9,129,22,39,253,19,98,108,110,79,113,224,232,
    178,185,112,104,218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,
    241,81,51,145,235,249,14,239,107,49,192,214,31,181,199,106,157,184,84,204,
    176,115,121,50,45,127,4,150,254,138,236,205,93,222,114,67,29,24,72,243,141,
    128,195,78,66,215,61,156,180], dtype=np.int64)

_P_PAPER = _PERMUTATION[np.arange(512) % 256]     # Ken Perlin 표준
_P_REPO = _PERMUTATION[np.arange(512) % 255]      # repo 의 버그 재현


def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _grad(h, x, y, z):
    h = h & 15
    u = np.where(h < 8, x, y)
    v = np.where(h < 4, y, np.where((h == 12) | (h == 14), x, z))
    return (np.where((h & 1) == 0, u, -u) + np.where((h & 2) == 0, v, -v))


def _perlin(x, y, z, P, unipolar):
    xi = np.floor(x).astype(np.int64) & 255
    yi = np.floor(y).astype(np.int64) & 255
    zi = np.floor(z).astype(np.int64) & 255
    xf, yf, zf = x - np.floor(x), y - np.floor(y), z - np.floor(z)
    u, v, w = _fade(xf), _fade(yf), _fade(zf)

    # repo 의 inc() 는 REPEAT=0 이라 단순 +1 이다 (경계에서 P 는 512 길이라 안전)
    aaa = P[P[P[xi] + yi] + zi]
    aba = P[P[P[xi] + yi + 1] + zi]
    aab = P[P[P[xi] + yi] + zi + 1]
    abb = P[P[P[xi] + yi + 1] + zi + 1]
    baa = P[P[P[xi + 1] + yi] + zi]
    bba = P[P[P[xi + 1] + yi + 1] + zi]
    bab = P[P[P[xi + 1] + yi] + zi + 1]
    bbb = P[P[P[xi + 1] + yi + 1] + zi + 1]

    x1 = _fade_lerp(_grad(aaa, xf, yf, zf), _grad(baa, xf - 1, yf, zf), u)
    x2 = _fade_lerp(_grad(aba, xf, yf - 1, zf), _grad(bba, xf - 1, yf - 1, zf), u)
    y1 = _fade_lerp(x1, x2, v)
    x1 = _fade_lerp(_grad(aab, xf, yf, zf - 1), _grad(bab, xf - 1, yf, zf - 1), u)
    x2 = _fade_lerp(_grad(abb, xf, yf - 1, zf - 1), _grad(bbb, xf - 1, yf - 1, zf - 1), u)
    y2 = _fade_lerp(x1, x2, v)
    out = _fade_lerp(y1, y2, w)
    return (out + 1.0) / 2.0 if unipolar else out


def _fade_lerp(a, b, t):
    return a + t * (b - a)


def _cmap(v, phi, kind):
    """색채맵. 논문 Eq.2 는 sin 하나뿐이고, 나머지는 확장축이다 (표에 별도 행으로 표기)."""
    a = v * 2.0 * np.pi * phi
    if kind == "sin":
        return np.sin(a)
    if kind == "tri":                       # 삼각파 — sin 과 같은 주기, 더 평평한 봉우리
        return 2.0 / np.pi * np.arcsin(np.sin(a))
    if kind == "saw":                       # 톱니 — 불연속, 고주파 성분 최대
        return 2.0 * ((a / (2.0 * np.pi)) % 1.0) - 1.0
    if kind == "square":                    # 구형파 — 부호 패턴에 가장 가깝다
        return np.sign(np.sin(a))
    if kind == "tanh":                      # tanh(k·sin) — sin 과 square 사이
        return np.tanh(3.0 * np.sin(a))
    raise ValueError(kind)


def generate_noise(T, frame_height, frame_width, num_octaves, wavelength_x,
                   wavelength_y, wavelength_t, color_period, epsilon,
                   mode="paper", renorm=False, offset=(0.0, 0.0, 0.0),
                   colormap="sin", persistence=1.0, channel_seeds=None):
    """(T,H,W) float64 노이즈. mode="repo" 는 Rust 구현과 동일한 결과를 낸다.

    mode="paper" 에서 num_octaves 는 논문의 Lambda 이며 l=0..Lambda 의 Lambda+1 항을 더한다.
    color_period 는 논문의 phi 이고 cmap(p,phi)=sin(p*2*pi*phi) 로 쓰인다.
    mode="repo" 에서는 repo 인자 규약 그대로 (num_octaves 항, sin(v*2pi/color_period)).
    """
    paper = (mode == "paper")
    P = _P_PAPER if paper else _P_REPO
    ox, oy, ot = offset
    t = (np.arange(T, dtype=np.float64) + ot)[:, None, None]
    y = (np.arange(frame_height, dtype=np.float64) + oy)[None, :, None]
    x = (np.arange(frame_width, dtype=np.float64) + ox)[None, None, :]

    n_terms = int(num_octaves) + 1 if paper else int(num_octaves)

    def _field(ox2, oy2, ot2):
        xx = x + ox2; yy = y + oy2; tt = t + ot2
        out = np.zeros((T, frame_height, frame_width), dtype=np.float64)
        for l in range(n_terms):
            f = 2.0 ** l
            out += (persistence ** l) * _perlin(
                xx * (f / wavelength_x), yy * (f / wavelength_y),
                tt * (f / wavelength_t), P, unipolar=not paper)
        return out

    phi = color_period if paper else 1.0 / max(color_period, 1e-12)
    if channel_seeds is None:                      # 논문 규약: 3채널 공통
        noise = _cmap(_field(0.0, 0.0, 0.0), phi, colormap) * epsilon
    else:                                          # 확장축: 채널별 독립 실현
        noise = np.stack([_cmap(_field(*s3), phi, colormap) for s3 in channel_seeds]) * epsilon
    if renorm:
        mx = np.abs(noise).max()
        if mx > 1e-12:
            noise = noise * (epsilon / mx)
    return noise


# ------------------------------------------------------------------ GPU (torch) 구현
# 위 numpy 구현과 같은 값을 낸다 (selftest 에서 1e-12 이내로 대조). PSO 적합도 평가마다 노이즈를
# 새로 합성해야 하는 딥페이크 파이프라인(deepfake/)이 쓴다.
def _perlin_t(x, y, z, P, unipolar):
    import torch
    xi, yi, zi = (torch.floor(x).long() & 255), (torch.floor(y).long() & 255), (torch.floor(z).long() & 255)
    xf, yf, zf = x - torch.floor(x), y - torch.floor(y), z - torch.floor(z)
    u, v, w = _fade(xf), _fade(yf), _fade(zf)

    def grad(h, a, b, c):
        h = h & 15
        p = torch.where(h < 8, a, b)
        q = torch.where(h < 4, b, torch.where((h == 12) | (h == 14), a, c))
        return torch.where((h & 1) == 0, p, -p) + torch.where((h & 2) == 0, q, -q)

    aaa = P[P[P[xi] + yi] + zi]
    aba = P[P[P[xi] + yi + 1] + zi]
    aab = P[P[P[xi] + yi] + zi + 1]
    abb = P[P[P[xi] + yi + 1] + zi + 1]
    baa = P[P[P[xi + 1] + yi] + zi]
    bba = P[P[P[xi + 1] + yi + 1] + zi]
    bab = P[P[P[xi + 1] + yi] + zi + 1]
    bbb = P[P[P[xi + 1] + yi + 1] + zi + 1]
    x1 = _fade_lerp(grad(aaa, xf, yf, zf), grad(baa, xf - 1, yf, zf), u)
    x2 = _fade_lerp(grad(aba, xf, yf - 1, zf), grad(bba, xf - 1, yf - 1, zf), u)
    y1 = _fade_lerp(x1, x2, v)
    x1 = _fade_lerp(grad(aab, xf, yf, zf - 1), grad(bab, xf - 1, yf, zf - 1), u)
    x2 = _fade_lerp(grad(abb, xf, yf - 1, zf - 1), grad(bbb, xf - 1, yf - 1, zf - 1), u)
    y2 = _fade_lerp(x1, x2, v)
    out = _fade_lerp(y1, y2, w)
    return (out + 1.0) / 2.0 if unipolar else out


def generate_noise_torch(T, frame_height, frame_width, num_octaves, wavelength_x, wavelength_y,
                         wavelength_t, color_period, epsilon, mode="paper", colormap="sin",
                         device="cuda"):
    """generate_noise 와 같은 규약의 (T,H,W) float64 torch 텐서. colormap 은 sin / square 만."""
    import torch
    paper = (mode == "paper")
    P = torch.as_tensor(_P_PAPER if paper else _P_REPO, dtype=torch.long, device=device)
    t, y, x = torch.meshgrid(torch.arange(T, dtype=torch.float64, device=device),
                             torch.arange(frame_height, dtype=torch.float64, device=device),
                             torch.arange(frame_width, dtype=torch.float64, device=device),
                             indexing="ij")
    n_terms = int(round(float(num_octaves))) + 1 if paper else int(round(float(num_octaves)))
    out = torch.zeros_like(x)
    for l in range(n_terms):
        f = 2.0 ** l
        out = out + _perlin_t(x * (f / wavelength_x), y * (f / wavelength_y),
                              t * (f / wavelength_t), P, unipolar=not paper)
    phi = color_period if paper else 1.0 / max(color_period, 1e-12)
    a = torch.sin(out * 2.0 * np.pi * phi)
    if colormap == "square":
        a = torch.sign(a)
    elif colormap != "sin":
        raise ValueError(colormap)
    return a * epsilon
