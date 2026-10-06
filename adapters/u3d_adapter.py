#!/usr/bin/env python
"""U3D (Universal 3-Dimensional perturbations, IEEE S&P'22) vs the shared C3D.

Two phases, both resumable:

  1. OPTIMISE - upstream `u3d-attack-C3D.py:attack_objective` is imported and
     driven by the upstream Rust PSO (`psolib.particle_swarm_optimization`)
     over the five Perlin-noise parameters
     [num_octaves, wavelength_x, wavelength_y, wavelength_t, color_period],
     with the repo's own bounds and PSO settings (swarmsize=20, omega=1.2,
     phip=phig=2.0, maxiter=40, T=16, alpha=0.5, I=5).  The result is cached to
     results/u3d/u3d_params.json so a crash never repeats the search.

  2. EVALUATE - the resulting universal perturbation is added to every video of
     the frozen evaluation subset, exactly as
     `noise_perturbation/perlin_noise.py:add_perlin_noise_to_frame` does
     (same value on all three channels, then clamp to [0,255]), and scored by
     the shared harness.

Threat model note: this third-party implementation's PSO objective maximises a
power-normalised distance between the victim's *intermediate features*, i.e. it
uses white-box feature access to the same network it attacks. The original paper
obtains those features from a surrogate and only queries the target (docs/U3D.md
§4.3); the perturbation itself is still universal and input-agnostic, so QS/QT
remain "-".

Noise variants (--noise), all searched by the same upstream objective and PSO:

  repo     upstream Rust generator, unchanged (default; faithful reproduction)
  paper    paper Eq. 1-2 exactly (dfbench.u3d_perlin, mode="paper")
  nyquist  U3D-N: the paper noise used as a smooth envelope on Nyquist carriers,
           eps * sign((-1)^y * N_p(x,y,t) + beta * (-1)^x); beta is a 6th PSO dim

For paper / nyquist the upstream `attack_objective` is used as is: only the
module-level `generate_noise` it calls is swapped at runtime, so no file under
repos/u3d is modified.  --query_weight adds the paper's Eq. 13 query term.
Why each of these exists, with measurements, is in docs/U3D.md §4 and §6.

The UAP is optimised on TRAINING videos, disjoint from the test videos it is
evaluated on.
"""
import argparse, importlib.util, json, os, sys, time
import numpy as np
import torch

# 프로젝트 루트: DF_ROOT 우선 (Jetson 은 ~/bench 로 경로가 다르다)
DF = os.environ.get("DF_ROOT",
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
U3D = os.path.join(DF, "repos", "u3d")
sys.path.insert(0, os.path.join(DF, "common"))
sys.path.insert(0, os.path.join(U3D, "python"))

from dfbench.runner import AttackRunner, DEFAULT_EPS                      # noqa: E402
from dfbench.paths import DATASET_DIR, RESULT_DIR                         # noqa: E402
from dfbench.victim import C3DVictim                                      # noqa: E402
from dfbench.records import atomic_write_json                             # noqa: E402
from dfbench import u3d_perlin                                            # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "u3d_attack_c3d", os.path.join(U3D, "python", "u3d-attack-C3D.py"))
UP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(UP)
UP.device = torch.device("cuda:0")          # repo hardcodes cuda:2 (multi-GPU box)

from psolib import particle_swarm_optimization as pso                     # noqa: E402

PARAMS_PATH = os.path.join(RESULT_DIR, "u3d", "u3d_params.json")


# --------------------------------------------------------------- 논문 정합 보정
# Xie et al., S&P'22, Eq.1 / Eq.2:
#     N(x,y,t) = sum_{l=0}^{Lambda} p(x*2^l/lx, y*2^l/ly, t*2^l/lt)      (Eq.1)
#     N_p      = cmap(N, phi),   cmap(p, phi) = sin(p * 2*pi*phi)        (Eq.2)
#
# 이 저장소의 Rust 구현은 두 곳에서 어긋난다
#   rust/perlin/src/perlin3d.rs:157   sin(value * 2*pi / color_period)   <- 나누기
#   rust/perlin/src/perlin3d.rs:175   for l in 0..num_octaves            <- Lambda 항
# Eq.2 는 곱하기이고 Eq.1 은 l=0..Lambda 즉 Lambda+1 항이다.
# Rust 를 재컴파일하지 않고 동일한 결과를 얻으려면 인자만 바꿔 넘기면 된다:
#   color_period <- 1/phi      =>  sin(N * 2*pi / (1/phi)) = sin(N * 2*pi*phi)
#   num_octaves  <- Lambda+1
def gen_noise(UPmod, T, num_octaves, wl_x, wl_y, wl_t, phi, eps, paper_cmap=True):
    if paper_cmap:
        return UPmod.generate_noise(T=T, frame_height=112, frame_width=112,
                                    num_octaves=int(num_octaves) + 1,
                                    wavelength_x=wl_x, wavelength_y=wl_y,
                                    wavelength_t=wl_t,
                                    color_period=1.0 / max(float(phi), 1e-9),
                                    epsilon=eps)
    return UPmod.generate_noise(T=T, frame_height=112, frame_width=112,
                                num_octaves=int(num_octaves),
                                wavelength_x=wl_x, wavelength_y=wl_y,
                                wavelength_t=wl_t, color_period=phi, epsilon=eps)

def _file_tag(a):
    tag = "" if abs(a.eps - 10) < 1e-6 else f"_e{a.eps:g}"
    if a.paper_cmap:
        tag += "_paper"
    return tag + a.tag


# 캐시는 실행 이름(--name) 아래에 둔다. 예전에는 이름과 무관하게 results/u3d/ 였기 때문에
# --name 을 바꿔 다른 설정을 돌려도 기본 실행의 파라미터를 그대로 재사용해 탐색을 건너뛰었다.
def params_path(a):
    return os.path.join(RESULT_DIR, a.name, f"u3d_params{_file_tag(a)}.json")


def perturbation_path(a):
    return os.path.join(RESULT_DIR, a.name, f"u3d_perturbation{_file_tag(a)}.npy")


def run_config(a):
    """캐시된 파라미터가 같은 설정에서 나왔는지 확인할 때 비교하는 항목."""
    return {"noise": a.noise, "colormap": a.colormap, "paper_cmap": a.paper_cmap,
            "fitness": a.fitness, "query_weight": a.query_weight, "epsilon": a.eps, "T": a.T,
            "restarts": a.restarts, "lb": list(a.lb), "ub": list(a.ub)}


# --------------------------------------------------------------- 노이즈 변형
# Eq.13 에서 업스트림 특징 거리(약 5e4)를 질의 성공률([0,1])과 같은 자릿수로 맞추는 배율.
# 논문은 D 의 절대 크기를 밝히지 않아 omega=10 을 그대로 쓰면 질의 항이 사라진다.
D_SCALE = 1e4
_CY = (-1.0) ** np.arange(112)[None, :, None]       # 행 나이퀴스트 반송파 (-1)^y
_CX = (-1.0) ** np.arange(112)[None, None, :]       # 열 나이퀴스트 반송파 (-1)^x


def synth(a, p):
    """PSO 파라미터 -> (T,112,112) 섭동. 어느 변형이든 |.| <= eps."""
    p = list(p)
    p[0] = int(round(p[0]))
    if a.noise == "repo":
        return gen_noise(UP, a.T, p[0], p[1], p[2], p[3], p[4], a.eps, paper_cmap=a.paper_cmap)
    if a.noise == "paper":
        return u3d_perlin.generate_noise(a.T, 112, 112, p[0], p[1], p[2], p[3], p[4], a.eps,
                                         mode="paper", colormap=a.colormap)
    # nyquist (U3D-N): 논문 노이즈를 [-1,1] 포락선으로 만들어 행 반송파에 싣고 열 반송파를 더한다
    env = u3d_perlin.generate_noise(a.T, 112, 112, p[0], p[1], p[2], p[3], p[4], 1.0,
                                    mode="paper")
    return np.where(_CY * env + p[5] * _CX >= 0, a.eps, -a.eps)


def patch_upstream_noise(a, extra):
    """업스트림 attack_objective 가 내부에서 부르는 generate_noise 를 synth 로 바꿔 끼운다.
    repos/u3d 의 파일은 건드리지 않는다 — 모듈 전역 이름만 교체한다.
    extra: 업스트림 시그니처에 없는 추가 파라미터 (nyquist 의 beta)."""
    def generate_noise(T, frame_height, frame_width, num_octaves, wavelength_x,
                       wavelength_y, wavelength_t, color_period, epsilon):
        return synth(a, [num_octaves, wavelength_x, wavelength_y, wavelength_t,
                         color_period] + list(extra))
    UP.generate_noise = generate_noise


class QuerySet:
    """논문 Eq.13 의 질의 집합 Q: 피해 모델이 맞힌 train 클립 n 개.
    order 는 train 인덱스 순열, used 는 다른 집합이 이미 가져간 인덱스 (여기서 갱신된다).
    같은 순열·used 를 넘기면 여러 집합이 서로 겹치지 않는다."""
    def __init__(self, victim, n, order, used):
        mm = np.load(os.path.join(DATASET_DIR, "train_clips_u8.npy"), mmap_mode="r")
        lab = np.array([int(l.rstrip("\n").split("\t")[2]) for l in
                        open(os.path.join(DATASET_DIR, "train_clips_index.txt"))])
        xs, ys, picked = [], [], []
        for s in range(0, len(order), 64):
            idx = np.sort([i for i in order[s:s + 64] if i not in used])
            if len(idx) == 0:
                continue
            x = torch.from_numpy(np.array(mm[idx], dtype=np.float32)).to(victim.device)
            y = torch.from_numpy(lab[idx]).to(victim.device)
            keep = victim.predict(x).argmax(1) == y
            xs.append(x[keep]); ys.append(y[keep]); picked.extend(idx[keep.cpu().numpy()])
            if len(picked) >= n:
                break
        self.x, self.y = torch.cat(xs)[:n], torch.cat(ys)[:n]
        used.update(int(i) for i in picked[:n])

    @torch.no_grad()
    def true_prob(self, victim, noise, shifts):
        """평균 정답 확률. 성공률(Q)의 연속 대리 — 질의 집합이 작아도 순위가 매끄럽다."""
        n = torch.as_tensor(noise, device=victim.device, dtype=torch.float32)
        s_sum = 0.0
        for s in shifts:
            ns = torch.roll(n, int(s), dims=0)[None, None]
            for i in range(0, len(self.x), 32):
                pr = torch.softmax(victim.predict((self.x[i:i + 32] + ns).clamp(0, 255)), 1)
                s_sum += float(pr.gather(1, self.y[i:i + 32, None]).sum())
        return s_sum / (len(shifts) * len(self.x))

    @torch.no_grad()
    def success_rate(self, victim, noise, shifts):
        n = torch.as_tensor(noise, device=victim.device, dtype=torch.float32)
        wrong = 0
        for s in shifts:
            ns = torch.roll(n, int(s), dims=0)[None, None]           # (1,1,T,H,W)
            for i in range(0, len(self.x), 32):
                adv = (self.x[i:i + 32] + ns).clamp(0, 255)
                wrong += int((victim.predict(adv).argmax(1) != self.y[i:i + 32]).sum())
        return wrong / (len(shifts) * len(self.x))


class FeatureModel:
    """What UP.intermediate_features() calls: model.forward(x) -> (logits, dict)."""
    def __init__(self, victim):
        self.victim = victim

    def forward(self, x):
        return self.victim.features(x)

    def __call__(self, x):
        return self.forward(x)


class TrainClipLoader:
    """Yields (clip_batch, label) batches of pixel-space training clips.

    sample="class_prop" reproduces upstream `attack_data_prep.py:68-74` —
    per class take int(n_class/total * n_videos) clips, i.e. keep the class
    distribution of the source set.  That is what README Step 8 feeds the
    attack (default -n 500).  sample="random" is uniform over all train clips.
    """
    def __init__(self, n_videos, batch_size, seed=0, sample="random"):
        mm = np.load(os.path.join(DATASET_DIR, "train_clips_u8.npy"), mmap_mode="r")
        rng = np.random.RandomState(seed)
        if sample == "class_prop":
            lab = np.array([int(l.rstrip("\n").split("\t")[2]) for l in
                            open(os.path.join(DATASET_DIR, "train_clips_index.txt"))])
            total, sel = len(lab), []
            for c in np.unique(lab):
                pool = np.where(lab == c)[0]
                k = int(len(pool) / total * n_videos)
                if k:
                    sel.extend(rng.choice(pool, k, replace=False))
            self.idx = np.array(sel)
        else:
            self.idx = rng.choice(mm.shape[0], n_videos, replace=False)
        self.mm, self.bs = mm, batch_size

    def __len__(self):
        return int(np.ceil(len(self.idx) / self.bs))

    def __iter__(self):
        for i in range(0, len(self.idx), self.bs):
            chunk = np.array(self.mm[np.sort(self.idx[i:i + self.bs])], dtype=np.float32)
            yield torch.from_numpy(chunk), torch.zeros(len(chunk), dtype=torch.long)


def optimise(a, victim):
    P = params_path(a)
    if os.path.exists(P) and not a.force_optimise:
        d = json.load(open(P))
        cached = d.get("config", {"noise": "repo"})
        mismatch = {k: (cached.get(k), v) for k, v in run_config(a).items()
                    if k in cached and cached.get(k) != v}
        if mismatch:
            raise SystemExit(f"[u3d] {P} was produced with a different configuration "
                             f"{mismatch}. Use another --name or pass --force_optimise.")
        print(f"[u3d] reusing cached PSO result: {d['best_params']}", flush=True)
        return d["best_params"]

    np.random.seed(a.seed)                    # 업스트림 목적함수의 tau 추출을 재현 가능하게
    model = FeatureModel(victim)
    loader = TrainClipLoader(a.n_opt_videos, a.opt_batch, seed=a.seed,
                             sample=a.sample)
    need_q = a.query_weight > 0 or a.fitness == "score"
    restarts = max(1, a.restarts)
    # train 클립 순열 하나에서 질의 집합(재시작마다 하나)과 선택 집합을 겹치지 않게 떼어 낸다
    order = np.random.RandomState(a.seed + 1000).permutation(
        len(open(os.path.join(DATASET_DIR, "train_clips_index.txt")).readlines()))
    used = set()
    qsets = [QuerySet(victim, a.n_query, order, used) if need_q else None
             for _ in range(restarts)]
    sel = QuerySet(victim, a.n_select, order, used) if restarts > 1 else None
    ns = argparse.Namespace(T=a.T, epsilon=a.eps, alpha=a.alpha, iteration=a.iteration)
    n_eval = [0]
    t0 = time.time()

    def make_objective(qset):
        def objective(params):
            params = [round(p) if i == 0 else p for i, p in enumerate(params)]
            if a.fitness == "score":          # Eq.13 의 Q 만, 성공률 대신 연속 대리 1 - p_true
                fit = 1.0 - qset.true_prob(victim, synth(a, params), (0, a.T // 2))
                msg = f"1-p_true={fit:.4f}"
            else:
                core = params[:5]
                if a.noise != "repo":         # 업스트림 목적함수가 synth 로 노이즈를 만들게 한다
                    patch_upstream_noise(a, params[5:])
                elif a.paper_cmap:            # Eq.2 부호 + Eq.1 항 수 보정 (Rust 인자 변환)
                    core = [core[0] + 1, core[1], core[2], core[3], 1.0 / max(core[4], 1e-9)]
                d = UP.attack_objective(ns, model, loader, core)
                fit, msg = d, f"dist={d:.1f}"
                if a.query_weight > 0:        # 논문 Eq.13: D + omega * Q
                    q = qset.success_rate(victim, synth(a, params), (0, a.T // 2))
                    fit = d / D_SCALE + a.query_weight * q
                    msg += f"  Q={q:.3f}  fit={fit:.3f}"
            n_eval[0] += 1
            if n_eval[0] % 10 == 0:
                el = time.time() - t0
                tot = restarts * a.swarmsize * (a.maxiter + 1)
                print(f"[u3d-pso] eval {n_eval[0]}/~{tot}  {msg}  "
                      f"elapsed {el/60:.1f}m  ETA {(tot-n_eval[0])*el/max(n_eval[0],1)/60:.1f}m",
                      flush=True)
            return -fit
        return objective

    lb, ub = list(a.lb), list(a.ub)
    names = ["num_octaves", "wavelength_x", "wavelength_y", "wavelength_t", "color_period"]
    if a.noise == "nyquist":
        lb.append(a.beta_bounds[0]); ub.append(a.beta_bounds[1]); names.append("beta")
    print(f"[u3d] noise={a.noise}  fitness={a.fitness}  restarts={restarts}  "
          f"PSO: swarmsize={a.swarmsize} maxiter={a.maxiter} "
          f"omega={a.omega} (effective inertia {1 + a.omega:.2f}) "
          f"(~{restarts*a.swarmsize*(a.maxiter+1)} objective evaluations)", flush=True)
    runs = []
    for r in range(restarts):
        # 레포는 minstep/minfunc 를 넘기지 않아 Rust 기본값 1e-8 로 1 회차에 멈춘다.
        # 논문이 명시한 maxiter 를 실제로 돌리려면 두 값을 0 으로 내려야 한다.
        b, _ = pso(make_objective(qsets[r]), lb=np.array(lb, dtype=float),
                   ub=np.array(ub, dtype=float), swarmsize=a.swarmsize, omega=a.omega,
                   phip=a.phip, phig=a.phig, maxiter=a.maxiter,
                   minstep=a.minstep, minfunc=a.minfunc, debug=True)
        b = [int(round(p)) if i == 0 else float(p) for i, p in enumerate(list(b))]
        run = {"params": b}
        if sel is not None:                   # 질의 집합과 겹치지 않는 train 클립으로 고른다
            run["select_true_prob"] = sel.true_prob(victim, synth(a, b), (0, a.T // 2))
        runs.append(run)
        print(f"[u3d] restart {r + 1}/{restarts}: {b}"
              + (f"  selection p_true={run['select_true_prob']:.4f}" if sel else ""), flush=True)
    pick = min(range(restarts), key=lambda r: runs[r].get("select_true_prob", 0.0))
    best = runs[pick]["params"]
    atomic_write_json(P, {
        "best_params": best,
        "param_names": names,
        "config": run_config(a),
        "pso": {"swarmsize": a.swarmsize, "omega": a.omega, "phip": a.phip,
                "phig": a.phig, "maxiter": a.maxiter, "lb": lb, "ub": ub},
        "epsilon": a.eps, "alpha": a.alpha, "iteration": a.iteration, "T": a.T,
        "n_opt_videos": a.n_opt_videos, "n_query": a.n_query if need_q else 0,
        "restarts": runs, "picked": pick, "n_select": a.n_select if sel else 0,
        "objective_evals": n_eval[0],
        "optimise_seconds": round(time.time() - t0, 1),
    })
    print(f"[u3d] best params {best}  ({(time.time()-t0)/60:.1f} min)", flush=True)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tier", type=int, default=1)
    p.add_argument("--eps", type=float, default=DEFAULT_EPS)
    p.add_argument("--lb", type=float, nargs=5, default=[1, 2.0, 2.0, 2.0, 1.0])
    p.add_argument("--ub", type=float, nargs=5, default=[5, 180.0, 180.0, 180.0, 60.0])
    p.add_argument("--swarmsize", type=int, default=20)
    p.add_argument("--omega", type=float, default=1.2,
                   help="passed to psolib unchanged. psolib updates v <- v + omega*v + ..., so "
                        "the effective inertia is 1+omega; 1.2 diverges (docs/U3D.md §4.5). "
                        "Use -0.27 for the standard 0.73")
    p.add_argument("--phip", type=float, default=2.0)
    p.add_argument("--phig", type=float, default=2.0)
    p.add_argument("--maxiter", type=int, default=40)
    p.add_argument("--minstep", type=float, default=0.0)
    p.add_argument("--minfunc", type=float, default=0.0)
    p.add_argument("--T", type=int, default=16)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--iteration", "-I", type=int, default=5)
    p.add_argument("--n_opt_videos", type=int, default=64)
    p.add_argument("--opt_batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sample", choices=["random", "class_prop"], default="random",
                   help="class_prop = README Step 8 (attack_data_prep.py) 방식")
    p.add_argument("--force_optimise", action="store_true")
    p.add_argument("--paper_cmap", type=lambda s: s == "True", default=False,
                   help="논문 Eq.1/Eq.2 로 보정 (sin(p*2pi*phi), Lambda+1 옥타브)")
    p.add_argument("--noise", choices=["repo", "paper", "nyquist"], default="repo",
                   help="repo = 업스트림 Rust 그대로, paper = 논문 Eq.1-2, "
                        "nyquist = U3D-N (논문 노이즈 포락선 x 나이퀴스트 반송파)")
    p.add_argument("--colormap", choices=["sin", "square"], default="sin",
                   help="--noise paper 의 색맵. square = sign(sin(.)), L-inf 예산을 꽉 채운다")
    p.add_argument("--beta_bounds", type=float, nargs=2, default=[0.0, 1.5],
                   help="--noise nyquist 의 열 반송파 비중 beta 탐색 범위")
    p.add_argument("--fitness", choices=["feature", "score"], default="feature",
                   help="feature = 업스트림 특징 거리 (--query_weight 로 Eq.13 질의 항 추가), "
                        "score = 질의 집합의 평균 정답 확률을 낮춘다 (Eq.13 의 Q 를 연속 대리로)")
    p.add_argument("--query_weight", type=float, default=0.0,
                   help="논문 Eq.13 의 omega. 0 이면 업스트림 목적함수만 쓴다")
    p.add_argument("--n_query", type=int, default=64,
                   help="Eq.13 질의 집합 크기 (피해 모델이 맞힌 train 클립)")
    p.add_argument("--restarts", type=int, default=1,
                   help="PSO 를 독립적으로 이만큼 돌리고, 질의 집합과 겹치지 않는 train 클립의 "
                        "평균 정답 확률로 하나를 고른다")
    p.add_argument("--n_select", type=int, default=256,
                   help="--restarts > 1 일 때 선택에 쓰는 train 클립 수")
    p.add_argument("--tag", default="", help="파라미터 캐시 파일 접미사")
    p.add_argument("--name", default="u3d")
    p.add_argument("--engine", default=None,
                   help="TensorRT 플랜(중간 특징 출력 포함). 주면 PSO 를 그 정밀도의 "
                        "엔진 위에서 돌린다 — 정밀도별 UAP 생성 비용 측정용.")
    p.add_argument("--eval_only", action="store_true",
                   help="PSO 만 하고 평가(AttackRunner)는 건너뛴다")
    a = p.parse_args()
    # 해당 노이즈에 의미가 없는 옵션은 조용히 무시하지 않고 멈춘다
    if a.paper_cmap and a.noise != "repo":
        p.error("--paper_cmap only applies to --noise repo; paper/nyquist already follow Eq. 1-2")
    if a.colormap != "sin" and a.noise != "paper":
        p.error("--colormap only applies to --noise paper")
    if a.fitness == "score" and a.query_weight > 0:
        p.error("--fitness score uses the query term alone; combine --query_weight with "
                "--fitness feature")

    os.makedirs(os.path.join(RESULT_DIR, a.name), exist_ok=True)
    if a.engine:
        from dfbench.trt_victim import TRTVictim
        victim = TRTVictim(a.engine, max_batch=16)
        if not victim.has_features:
            raise SystemExit(f"중간 특징을 내보내지 않는 엔진이다: {a.engine}")
        print(f"[u3d] victim = TensorRT {a.engine}", flush=True)
    else:
        victim = C3DVictim(max_batch=16)
    best = optimise(a, victim)
    del victim
    torch.cuda.empty_cache()

    noise = synth(a, best)
    np.save(perturbation_path(a), noise)
    print(f"[u3d] UAP generated: shape={noise.shape} "
          f"Linf={np.abs(noise).max():.3f} mean|.|={np.abs(noise).mean():.3f}", flush=True)

    def attack(victim, clean, label, eps, budget):
        # add_perlin_noise_to_frame: same noise on all channels, then clamp
        n = torch.as_tensor(noise, device=victim.device, dtype=torch.float32)
        adv = torch.as_tensor(clean, device=victim.device) + n.unsqueeze(0)
        adv = adv.clamp(0, 255)
        victim.note_iterate(adv.unsqueeze(0))
        return adv

    if a.eval_only:
        print("[u3d] --eval_only: 평가 생략", flush=True)
        return

    r = AttackRunner(a.name, tier=a.tier, eps=a.eps, budget=1)
    r.run(attack, note=f"universal {a.noise} U3D perturbation, PSO params={best}, "
                       f"eps={a.eps}, omega={a.omega}, fitness={a.fitness}, "
                       f"query_weight={a.query_weight}, restarts={a.restarts}, "
                       f"optimised on training clips only")


if __name__ == "__main__":
    main()
