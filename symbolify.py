# file: symbolify.py
#
# Post-training "symbolification" utilities for AdaptiveKANLayer / SimpleKAN.
# - Samples each learned 1-D edge spline y_{i->o}(x)
# - Fits a small primitive library g(x) = c * f(a * x + b) + d using LBFGS
# - If the fit is good (R^2 >= threshold), reprojects g onto the B-spline basis
#   and writes the coefficients back into layer.spline_coeffs[i, o, :]
# - Optionally freezes those coefficients by registering a gradient mask
# - Exports a JSON map of which edges were symbolified and with what params
#
# Usage (after training):
#     from symbolify import symbolify_model, attach_freeze_hooks
#     report = symbolify_model(model, out_json_path=os.path.join(out_dir, 'symbolic_map.json'),
#                              r2_threshold=0.995, replace=True, freeze=True)
#     attach_freeze_hooks(model)  # only if freeze=True above and you keep training
#
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    from spline import b_spline_basis
except Exception as e:
    raise ImportError("symbolify.py expects b_spline_basis in spline.py") from e

# ---- primitive function library ------------------------------------------------

class Primitive(nn.Module):
    def __init__(self, name: str, base: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.name = name
        # Parameters a,b,c,d are all learnable; initialized near identity
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))
        self.c = nn.Parameter(torch.tensor(1.0))
        self.d = nn.Parameter(torch.tensor(0.0))
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c * self.base(self.a * x + self.b) + self.d

    def init_from_data(self, x: torch.Tensor, y: torch.Tensor):
        with torch.no_grad():
            y_std = torch.std(y)
            if torch.isfinite(y_std) and y_std > 1e-6:
                self.c.copy_(y_std)
            self.d.copy_(torch.mean(y))
            self.a.copy_(torch.tensor(1.0))
            self.b.copy_(torch.mean(x) * 0.0)

    def params_dict(self):
        return {"a": float(self.a.detach().cpu()),
                "b": float(self.b.detach().cpu()),
                "c": float(self.c.detach().cpu()),
                "d": float(self.d.detach().cpu())}

def primitive_library() -> List[Primitive]:
    return [
        Primitive("identity", lambda z: z),
        Primitive("square",   lambda z: z**2),
        Primitive("exp",      torch.exp),
        Primitive("sin",      torch.sin),
        Primitive("cos",      torch.cos),
    ]

# ---- helpers -------------------------------------------------------------------

# --- Monitoring helpers ---
import time, logging, csv, os, subprocess

def _hms(seconds: float) -> str:
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def _gpu_util_mem():
    """Return (util%, mem_used_MB, mem_total_MB) or None if nvidia-smi not present."""
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits"
        ], stderr=subprocess.DEVNULL)
        rows = out.decode().strip().splitlines()
        idx = torch.cuda.current_device() if torch.cuda.is_available() else 0
        util, mem_used, mem_total = [int(x) for x in rows[idx].split(", ")]
        return util, mem_used, mem_total
    except Exception:
        return None


def _edge_response(layer, i_in: int, i_out: int, xs: torch.Tensor) -> torch.Tensor:
    """Evaluate one edge’s spline y_{i_in->i_out}(x) at points xs."""
    device = layer.spline_coeffs.device
    xs = xs.to(device=device, dtype=layer.spline_coeffs.dtype).view(-1, 1)
    knots = layer.grids[i_in]  # (n_knots,)
    B = b_spline_basis(xs, knots, int(layer.spline_degree))  # (N, n_basis)
    coeff = layer.spline_coeffs[i_in, i_out, :]              # (n_basis,)
    return B @ coeff

def _fit_one(xs: torch.Tensor, ys: torch.Tensor, primitive: Primitive,
             max_iter: int = 200) -> Tuple[float, float, Dict]:
    """Fit one primitive by MSE via LBFGS. Returns (mse, r2, params)."""
    device = ys.device
    xs = xs.to(device); ys = ys.to(device)

    model = primitive.to(device)
    model.init_from_data(xs, ys)

    def loss_fn(pred, target): return torch.mean((pred - target) ** 2)

    opt = torch.optim.LBFGS(model.parameters(), lr=0.8, max_iter=max_iter,
                             tolerance_change=1e-12, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(set_to_none=True)
        pred = model(xs)
        loss = loss_fn(pred, ys)
        loss.backward()
        return loss
    try:
        opt.step(closure)
    except Exception:
        opt2 = torch.optim.Adam(model.parameters(), lr=0.02)
        for _ in range(500):
            opt2.zero_grad(set_to_none=True)
            pred = model(xs); loss = loss_fn(pred, ys)
            loss.backward(); opt2.step()

    with torch.no_grad():
        pred = model(xs)
        mse = float(loss_fn(pred, ys).detach().cpu())
        var = float(torch.var(ys).detach().cpu())
        r2 = 1.0 - (mse / (var + 1e-12))
        params = model.params_dict()
    return mse, r2, {"name": model.name, **params}

def _project_to_spline(xs: torch.Tensor, ys: torch.Tensor,
                       knots: torch.Tensor, degree: int) -> torch.Tensor:
    """Solve least squares for coeffs c: B c ≈ ys. Returns c (n_basis,)."""
    B = b_spline_basis(xs.view(-1,1).to(knots.device), knots, int(degree)).to(ys.device)
    sol = torch.linalg.lstsq(B, ys)
    c = sol.solution.squeeze(-1)
    return c

def _edge_range(knots: torch.Tensor) -> Tuple[float, float]:
    return float(knots.min().item()), float(knots.max().item())

# ---- public API ----------------------------------------------------------------

@dataclass
class EdgeReport:
    layer: int
    i_in: int
    i_out: int
    range: Tuple[float, float]
    best_name: Optional[str] = None
    best_params: Optional[Dict] = None
    mse: Optional[float] = None
    r2: Optional[float] = None
    replaced: bool = False

def symbolify_model(model: nn.Module,
                    r2_threshold: float = 0.995,
                    replace: bool = True,
                    freeze: bool = False,
                    out_json_path: Optional[str] = None,
                    samples_per_edge: int = 400) -> Dict:
    """
    Iterate AdaptiveKANLayer-like modules in `model`, fit primitives to each edge spline,
    and (optionally) replace good fits by writing new spline coefficients.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    reports: List[EdgeReport] = []
    lib = primitive_library()

    for L, layer in enumerate(model.modules()):
        if not (hasattr(layer, "spline_coeffs") and hasattr(layer, "grids") and hasattr(layer, "spline_degree")):
            continue

        in_dim = layer.in_dim if hasattr(layer, "in_dim") else layer.spline_coeffs.shape[0]
        out_dim = layer.out_dim if hasattr(layer, "out_dim") else layer.spline_coeffs.shape[1]

        for i_in in range(in_dim):
            knots = layer.grids[i_in].detach()
            x_lo, x_hi = _edge_range(knots)
            xs = torch.linspace(x_lo, x_hi, samples_per_edge, device=device, dtype=dtype)
            for i_out in range(out_dim):
                with torch.no_grad():
                    ys = _edge_response(layer, i_in, i_out, xs)

                best = None
                for prim_proto in lib:
                    prim = Primitive(prim_proto.name, prim_proto.base)
                    mse, r2, params = _fit_one(xs, ys, prim, max_iter=200)
                    if (best is None) or (mse < best["mse"]):
                        best = {"mse": mse, "r2": r2, "name": params["name"], "params": params}

                rep = EdgeReport(layer=L, i_in=i_in, i_out=i_out, range=(x_lo, x_hi),
                                 best_name=best["name"], best_params=best["params"],
                                 mse=best["mse"], r2=best["r2"], replaced=False)

                if replace and best["r2"] >= r2_threshold:
                    with torch.no_grad():
                        name = best["name"]; p = best["params"]
                        base = {"identity": (lambda z: z),
                                "square": (lambda z: z**2),
                                "exp": torch.exp, "sin": torch.sin, "cos": torch.cos}[name]
                        a = torch.tensor(p["a"], device=device, dtype=dtype)
                        b = torch.tensor(p["b"], device=device, dtype=dtype)
                        c = torch.tensor(p["c"], device=device, dtype=dtype)
                        d = torch.tensor(p["d"], device=device, dtype=dtype)
                        gxs = c * base(a * xs + b) + d

                        coeff = _project_to_spline(xs, gxs, knots.to(device), int(layer.spline_degree))
                        layer.spline_coeffs.data[i_in, i_out, :coeff.numel()] = coeff
                        if layer.spline_coeffs.shape[-1] > coeff.numel():
                            layer.spline_coeffs.data[i_in, i_out, coeff.numel():] = 0.0

                        if freeze:
                            if not hasattr(layer, "freeze_mask"):
                                mask = torch.ones_like(layer.spline_coeffs.data)
                                layer.register_buffer("freeze_mask", mask)
                            layer.freeze_mask.data[i_in, i_out, :] = 0.0
                        rep.replaced = True

                reports.append(rep)

    result = {"r2_threshold": r2_threshold,
              "replaced": sum(1 for r in reports if r.replaced),
              "total_edges": len(reports),
              "edges": [asdict(r) for r in reports]}

    if out_json_path is not None:
        import os
        os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
        with open(out_json_path, "w") as f:
            json.dump(result, f, indent=2)

    return result

def attach_freeze_hooks(model: nn.Module):
    """
    If layers have a `freeze_mask` buffer, attach backward hooks that
    multiply grads by the mask so frozen edges stay frozen.
    Call this once after creating/updating freeze_mask.
    """
    for layer in model.modules():
        if hasattr(layer, "spline_coeffs") and hasattr(layer, "freeze_mask"):
            mask = layer.freeze_mask
            param = layer.spline_coeffs
            param.register_hook(lambda g, m=mask: g * m)


# --- pruning utilities ---------------------------------------------------------
from dataclasses import dataclass
from typing import Dict, List

def prune_edges(model,
                rel_thresh: float = 1e-2,   # prune edges with energy < rel_thresh * max_energy_per_output
                abs_thresh: float = None,   # or prune if absolute energy < abs_thresh
                samples_per_edge: int = 400,
                freeze: bool = True) -> Dict:
    """
    Structured pruning for KAN edges.
    For each AdaptiveKANLayer and each output unit, compute the RMS energy of every incoming edge
    over the layer's input range. Prune edges whose energy is small (relative or absolute).
    Pruning = set spline_coeffs[...] = 0. If freeze=True, also zero the grad mask for those edges.
    """
    import torch
    from spline import b_spline_basis

    reports: List[Dict] = []
    total = 0
    pruned = 0

    for L, layer in enumerate(model.modules()):
        if not (hasattr(layer, "spline_coeffs") and hasattr(layer, "grids") and hasattr(layer, "spline_degree")):
            continue

        device = layer.spline_coeffs.device
        dtype = layer.spline_coeffs.dtype
        in_dim = getattr(layer, "in_dim", layer.spline_coeffs.shape[0])
        out_dim = getattr(layer, "out_dim", layer.spline_coeffs.shape[1])

        # Precompute bases per input dim
        bases = []
        for i_in in range(in_dim):
            knots = layer.grids[i_in]
            xlo, xhi = float(knots.min()), float(knots.max())
            xs = torch.linspace(xlo, xhi, samples_per_edge, device=device, dtype=dtype).view(-1, 1)
            B = b_spline_basis(xs, knots, int(layer.spline_degree)).to(device)
            bases.append(B)

        # Energies and max per output
        energies = torch.zeros(in_dim, out_dim, device=device, dtype=dtype)
        max_energy_out = torch.zeros(out_dim, device=device, dtype=dtype)

        for i_in in range(in_dim):
            B = bases[i_in]                                    # (N, n_basis)
            coeffs = layer.spline_coeffs[i_in]                 # (out_dim, n_basis)
            ys = B @ coeffs.T                                  # (N, out_dim)
            e = torch.sqrt(torch.mean(ys**2, dim=0))           # (out_dim,)
            energies[i_in] = e
            max_energy_out = torch.maximum(max_energy_out, e)

        # Decide and apply pruning
        for i_out in range(out_dim):
            max_e = float(max_energy_out[i_out].item() + 1e-12)
            for i_in in range(in_dim):
                total += 1
                e = float(energies[i_in, i_out].item())
                rel_e = e / max_e
                do_prune = (rel_e < rel_thresh) or (abs_thresh is not None and e < abs_thresh)
                if do_prune:
                    with torch.no_grad():
                        layer.spline_coeffs.data[i_in, i_out, :] = 0.0
                        if freeze:
                            if not hasattr(layer, "freeze_mask"):
                                mask = torch.ones_like(layer.spline_coeffs.data)
                                layer.register_buffer("freeze_mask", mask)
                            layer.freeze_mask.data[i_in, i_out, :] = 0.0
                    pruned += 1
                reports.append(dict(layer=L, i_in=i_in, i_out=i_out,
                                    energy=e, max_energy_out=max_e, rel_energy=rel_e,
                                    pruned=bool(do_prune)))

    return {"total_edges": total, "pruned": pruned, "rel_thresh": rel_thresh,
            "abs_thresh": abs_thresh, "edges": reports}


# === Post-symbolify polish helpers ============================================
import os, json
from typing import Tuple, Optional, Dict
from dataclasses import asdict
import torch
import torch.nn as nn
import torch.nn.functional as F

def _reg_terms_from_splines(model: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute L1 and an entropy-like term directly from spline coefficients,
    without assuming any custom model APIs.
    """
    l1 = torch.tensor(0.0, device=next(model.parameters()).device)
    ent = torch.tensor(0.0, device=next(model.parameters()).device)
    count_l = 0
    count_e = 0
    for layer in model.modules():
        if hasattr(layer, "spline_coeffs"):
            w = layer.spline_coeffs
            l1 = l1 + w.abs().mean()
            count_l += 1
            # entropy-ish: per-edge distribution over basis (on |coeff|)
            p = w.abs() + 1e-12
            p = p / p.sum(dim=-1, keepdim=True)
            ent = ent - (p * (p + 1e-12).log()).mean()
            count_e += 1
    if count_l == 0:
        return torch.tensor(0.0, device=l1.device), torch.tensor(0.0, device=l1.device)
    return l1 / max(1, count_l), ent / max(1, count_e)

def set_lock_grids(model: nn.Module, lock: bool = True) -> None:
    """
    Set a flag on all layers so update_grid() can skip moving knots after symbolify.
    You must patch AdaptiveKANLayer.update_grid (see instructions) to honor this flag.
    """
    for layer in model.modules():
        setattr(layer, "lock_grids", bool(lock))

@torch.no_grad()
def _save_model(model: nn.Module, dst_path: str):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    torch.save({"model_state": model.state_dict()}, dst_path)


def fine_tune_after_symbolify(
    model: nn.Module,
    train_inputs: torch.Tensor,
    train_labels: torch.Tensor,
    cfg: dict,
    log_dir: str
) -> Dict:
    """
    Config-driven fine-tune after symbolify. Keys (with defaults):
      ft_optimizer: "lbfgs" | "adam" | "adamw"   (default "lbfgs")
      ft_lr: 1.0 for LBFGS, 6e-4 for Adam/AdamW
      ft_steps: 200
      ft_batch_size: "full" or int
      ft_l1_weight: 1.5e-3
      ft_entropy_weight: 0.0
      ft_log_interval: 200
      ft_heartbeat_secs: 45.0
      ft_ema_beta: 0.995
      ft_target_mse: None (float to enable)
      ft_patience_updates: None (int to enable)
      ft_min_improve_rel: 1e-3
      ft_min_improve_abs: 5e-5
      ft_prune_rel_thresh: 0.02
      ft_second_symbolify: True
      ft_r2_threshold: 0.995
      ft_samples_per_edge: 400
    """
    from torch.utils.data import TensorDataset, DataLoader
    import torch.nn.functional as F

    device = next(model.parameters()).device
    model.train()

    # --- read config ---
    opt_name = str(cfg.get("ft_optimizer", "lbfgs")).lower()
    steps = int(cfg.get("ft_steps", 200))
    bscfg = cfg.get("ft_batch_size", "full")
    l1_weight = float(cfg.get("ft_l1_weight", 1.5e-3))
    entropy_weight = float(cfg.get("ft_entropy_weight", 0.0))
    lr = float(cfg.get("ft_lr", 1.0 if opt_name == "lbfgs" else 6e-4))

    LOG_INTERVAL_STEPS = int(cfg.get("ft_log_interval", 200))
    HEARTBEAT_SECS     = float(cfg.get("ft_heartbeat_secs", 45.0))
    EMA_BETA           = float(cfg.get("ft_ema_beta", 0.995))

    TARGET_MSE       = cfg.get("ft_target_mse", None)
    TARGET_MSE       = float(TARGET_MSE) if TARGET_MSE not in (None, 0, "0") else None
    PATIENCE_UPDATES = cfg.get("ft_patience_updates", None)
    PATIENCE_UPDATES = int(PATIENCE_UPDATES) if PATIENCE_UPDATES not in (None, 0, "0") else None
    MIN_IMPROVE_REL  = float(cfg.get("ft_min_improve_rel", 1e-3))
    MIN_IMPROVE_ABS  = float(cfg.get("ft_min_improve_abs", 5e-5))

    prune_rel_thresh   = float(cfg.get("ft_prune_rel_thresh", 0.02))
    second_symbolify   = bool(cfg.get("ft_second_symbolify", True))
    r2_threshold       = float(cfg.get("ft_r2_threshold", 0.995))
    samples_per_edge   = int(cfg.get("ft_samples_per_edge", 400))

    # --- data / full-batch support ---
    ds = TensorDataset(train_inputs.to(device), train_labels.to(device))
    if isinstance(bscfg, str) and bscfg.lower() == "full":
        bs = len(ds) if len(ds) > 0 else 1
    else:
        bs = max(1, min(int(bscfg), len(ds))) if len(ds) > 0 else 1
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=False)

    # --- optimizer ---
    ft_params = [p for p in model.parameters() if p.requires_grad]
    if opt_name == "lbfgs":
        optimizer = torch.optim.LBFGS(
            ft_params, lr=lr,
            max_iter=int(cfg.get("ft_lbfgs_max_iter", 20)),
            history_size=int(cfg.get("ft_lbfgs_history_size", 10)),
            line_search_fn="strong_wolfe"
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(ft_params, lr=lr)
    else:
        optimizer = torch.optim.Adam(ft_params, lr=lr)

    # --- monitoring ---
    csv_path = os.path.join(log_dir, "fine_tune_log.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["update","epoch","batch","loss","ema_loss","lr","elapsed_s","eta_s","gpu_util","gpu_mem_MB"])
    csv_file.flush()

    start_t = last_log_t = time.perf_counter()
    updates_per_epoch = max(len(dl), 1)
    total_updates = steps if opt_name == "lbfgs" else steps * updates_per_epoch
    u = 0
    ema_loss = None
    best_ema, best_u = float("inf"), 0
    last_nsmi_check = 0.0

    def _loss_on(x, y):
        pred = model(x)
        main = F.mse_loss(pred, y)
        l1, ent = _reg_terms_from_splines(model)
        return main + l1_weight * l1 + entropy_weight * ent

    logging.info(
        f"[FT] opt={opt_name} lr={lr} bs={bs} batches/epoch={updates_per_epoch} steps={steps} "
        f"log_every={LOG_INTERVAL_STEPS} heartbeat={HEARTBEAT_SECS:.0f}s ema_beta={EMA_BETA}"
    )

    early_stop = False

    if opt_name == "lbfgs":
        # Force full-batch for LBFGS
        total_updates = steps
        full_x = train_inputs.to(device)
        full_y = train_labels.to(device)

        for epoch in range(steps):
            def closure():
                optimizer.zero_grad(set_to_none=True)
                loss = _loss_on(full_x, full_y)
                loss.backward()
                return loss
            loss = optimizer.step(closure)
            cur = float(loss.detach().item())
            ema_loss = cur if ema_loss is None else (EMA_BETA * ema_loss + (1 - EMA_BETA) * cur)
            u += 1

            # --- early-stop metric (loss vs ema) ---
            TARGET_METRIC = str(cfg.get("ft_target_metric", "ema")).lower()  # "ema" | "loss"
            metric_val = ema_loss if TARGET_METRIC == "ema" else cur

            if u == 1:
                best_metric, best_u = metric_val, u
            else:
                improved = (best_metric - metric_val) > max(MIN_IMPROVE_ABS, best_metric * MIN_IMPROVE_REL)
                if improved:
                    best_metric, best_u = metric_val, u

            # stop early if target reached or patience exceeded
            if (TARGET_MSE is not None and metric_val <= TARGET_MSE) or (PATIENCE_UPDATES and (u - best_u) >= PATIENCE_UPDATES):
                early_stop = True
                break


            now = time.perf_counter()
            elapsed = now - start_t
            eta = (elapsed / u) * max(0, total_updates - u)
            lr_now = optimizer.param_groups[0]["lr"]

            if (u % LOG_INTERVAL_STEPS == 0) or ((now - last_log_t) >= HEARTBEAT_SECS) or u in (1, total_updates):
                logging.info(f"[FT] upd {u}/{total_updates} (LBFGS ep {epoch+1}/{steps}) "
                             f"loss={cur:.3e} ema={ema_loss:.3e} lr={lr_now:.2e} "
                             f"elapsed={_hms(elapsed)} eta={_hms(eta)}")
                csv_writer.writerow([u, epoch+1, 1, cur, ema_loss, lr_now, int(elapsed), int(eta), "", ""])
                csv_file.flush()
                last_log_t = now

            improved = (best_ema - ema_loss) > max(MIN_IMPROVE_ABS, best_ema * MIN_IMPROVE_REL)
            if improved:
                best_ema, best_u = ema_loss, u
            if (TARGET_MSE is not None and ema_loss <= TARGET_MSE) or (PATIENCE_UPDATES and (u - best_u) >= PATIENCE_UPDATES):
                early_stop = True
                break
    else:
        # Adam/AdamW over loader
        for epoch in range(steps):
            for bi, (xb, yb) in enumerate(dl, 1):
                loss = _loss_on(xb, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ft_params, float(cfg.get("ft_grad_clip", 1.0)))
                optimizer.step()

                u += 1
                cur = float(loss.detach().item())
                ema_loss = cur if ema_loss is None else (EMA_BETA * ema_loss + (1 - EMA_BETA) * cur)

                # --- early-stop metric (loss vs ema) ---
                TARGET_METRIC = str(cfg.get("ft_target_metric", "ema")).lower()  # "ema" | "loss"
                metric_val = ema_loss if TARGET_METRIC == "ema" else cur

                if u == 1:
                    best_metric, best_u = metric_val, u
                else:
                    improved = (best_metric - metric_val) > max(MIN_IMPROVE_ABS, best_metric * MIN_IMPROVE_REL)
                    if improved:
                        best_metric, best_u = metric_val, u

                # stop early if target reached or patience exceeded
                if (TARGET_MSE is not None and metric_val <= TARGET_MSE) or (PATIENCE_UPDATES and (u - best_u) >= PATIENCE_UPDATES):
                    early_stop = True
                    break


                now = time.perf_counter()
                elapsed = now - start_t
                eta = (elapsed / u) * max(0, total_updates - u)
                lr_now = optimizer.param_groups[0]["lr"]

                if (u % LOG_INTERVAL_STEPS == 0) or ((now - last_log_t) >= HEARTBEAT_SECS) or u in (1, total_updates):
                    logging.info(
                        f"[FT] upd {u}/{total_updates} (ep {epoch+1}/{steps}, b {bi}/{updates_per_epoch}) "
                        f"loss={cur:.3e} ema={ema_loss:.3e} lr={lr_now:.2e} "
                        f"elapsed={_hms(elapsed)} eta={_hms(eta)}"
                    )
                    csv_writer.writerow([u, epoch+1, bi, cur, ema_loss, lr_now, int(elapsed), int(eta), "", ""])
                    csv_file.flush()
                    last_log_t = now

                improved = (best_ema - ema_loss) > max(MIN_IMPROVE_ABS, best_ema * MIN_IMPROVE_REL)
                if improved:
                    best_ema, best_u = ema_loss, u
                if (TARGET_MSE is not None and ema_loss <= TARGET_MSE) or (PATIENCE_UPDATES and (u - best_u) >= PATIENCE_UPDATES):
                    early_stop = True
                    break
            if early_stop:
                break

    csv_file.close()
    if early_stop:
        logging.info(f"[FT] early-stopped at update {u}/{total_updates} with EMA={ema_loss:.3e} (best={best_ema:.3e} at upd {best_u}).")
    else:
        logging.info(f"[FT] done in {_hms(time.perf_counter() - start_t)}  (log: {csv_path})")

    # Optional prune
    prune_report = None
    if prune_rel_thresh and prune_rel_thresh > 0:
        if "prune_edges" in globals():
            prune_report = prune_edges(model, rel_thresh=prune_rel_thresh, freeze=True)  # type: ignore
        else:
            prune_report = {"warning": "prune_edges not found; skip pruning"}
        with open(os.path.join(log_dir, "prune_report_after_ft.json"), "w") as f:
            json.dump(prune_report, f, indent=2)

    # Optional second symbolify
    sym2 = None
    if second_symbolify:
        sym2 = symbolify_model(  # type: ignore
            model,
            r2_threshold=r2_threshold,
            replace=True,
            freeze=True,
            out_json_path=os.path.join(log_dir, "symbolic_map_final.json"),
            samples_per_edge=samples_per_edge,
        )

    # Save updated model
    _save_model(model, os.path.join(log_dir, "pykan_mit_model_after_ft.pth"))
    return {"prune_report": prune_report, "symbolic_report2": sym2}

