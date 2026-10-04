"""Fuse every available metric-scale cue into one scale factor with an honest interval (D-013).

Scale factor s: metric = s * reconstruction units. Every cue is an independent-ish estimate of s:
  * sheet      - camera-to-floor distance measured from a paper sheet / same distance in the reconstruction (strong,
                 ~0.5-2%); optional, off by default (D-067: no reference object in the protocol);
  * learned_depth - metric depth network vs. reconstruction depth (MoGe-2 / DA3 / MapAnything; a few percent, with an
                 EMPIRICAL sigma measured by the caller);
  * priors     - camera held at chest height (~1.3-1.6 m), interior door height (~2.0-2.1 m), ceiling height
                 (~2.4-3.0 m). Weak, but always available, so the pipeline never stops ("any picture in, results
                 out") and they arbitrate when two strong cues disagree.

Method (all in log-scale, because scale errors are multiplicative):
  1. Robust outlier rejection: leave-one-out test of each cue against the fusion of the others; the worst cue with
     |z| > z_reject is dropped, repeatedly, but only while at least 3 cues remain (with 2 cues there is no majority to
     say which one is wrong; we then inflate the interval instead).
  2. Correlated cues: cues that share an error source (several photos with the same camera focal error, several
     frames through the same depth network) carry `sigma_sys` plus a `group`. Within a group the random parts average
     down, the systematic part does NOT. Treating them as independent would claim sqrt(N) more precision than exists.
  3. Inverse-variance weighting across groups; then a Birge-ratio inflation sqrt(chi2/dof) when the surviving cues
     are mutually inconsistent beyond their stated sigmas (calibration over optimism).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Priors (1-sigma). Chosen wide on purpose: they must arbitrate, never dominate a measured cue.
CAMERA_HEIGHT_PRIOR = (1.40, 0.13)     # chest-height shots (protocol); adults hold phones ~1.15-1.65 m
DOOR_HEIGHT_PRIOR = (2.04, 0.05)       # 2.00-2.10 m standard leaves (EU 2.0-2.1, US 6'8" = 2.03, IN 2.1)
CEILING_HEIGHT_PRIOR = (2.65, 0.25)    # residential 2.4-3.1 m (the sample apartment is ~3.08 m)


@dataclass
class ScaleCue:
    name: str
    kind: str                       # sheet | learned_depth | prior_camera_height | prior_door | prior_ceiling | other
    scale: float                    # metric / reconstruction units
    sigma_rand: float               # 1-sigma RELATIVE error that is independent between cues
    sigma_sys: float = 0.0          # 1-sigma RELATIVE error shared by every cue of the same group
    group: str | None = None        # cues sharing a systematic error (e.g. "camera", "moge2")
    note: str = ""

    @property
    def sigma(self) -> float:
        return float(np.hypot(self.sigma_rand, self.sigma_sys))


# ----------------------------------------------------------------------------------------------- cue builders
def sheet_cue(meas, h_recon: float, h_recon_rel_sigma: float = 0.0, group: str = "camera", name: str = "") -> ScaleCue:
    """Sheet measurement (floorplan.scale.sheet.SheetMeasurement) + the same camera's height above the floor in the
    reconstruction. The focal-length part of the sheet error is shared by all photos of one camera -> sigma_sys."""
    return ScaleCue(name or "sheet", "sheet", meas.height_m / h_recon,
                    float(np.hypot(meas.sigma_rand_m / meas.height_m, h_recon_rel_sigma)),
                    meas.sigma_sys_m / meas.height_m, group,
                    f"h={meas.height_m:.3f} m, paper={meas.paper}, h_recon={h_recon:.4f}")


def learned_depth_cue(ratio: float, rel_sigma: float, name: str = "learned_depth", group: str | None = None,
                      rel_sigma_sys: float = 0.0) -> ScaleCue:
    """ratio = metric depth (network) / reconstruction depth, robustly averaged by the caller. rel_sigma must be the
    EMPIRICAL scale error of that network measured against LiDAR, not its self-reported confidence."""
    return ScaleCue(name, "learned_depth", ratio, rel_sigma, rel_sigma_sys, group)


def camera_height_prior(h_recon: float, prior=CAMERA_HEIGHT_PRIOR) -> ScaleCue:
    """h_recon: median camera height above the floor in reconstruction units (protocol: chest-height shots)."""
    return ScaleCue("prior_camera_height", "prior_camera_height", prior[0] / h_recon, prior[1] / prior[0],
                    note=f"{prior[0]}+-{prior[1]} m")


def door_height_prior(h_recon: float, prior=DOOR_HEIGHT_PRIOR) -> ScaleCue:
    return ScaleCue("prior_door_height", "prior_door", prior[0] / h_recon, prior[1] / prior[0],
                    note=f"{prior[0]}+-{prior[1]} m")


def ceiling_height_prior(h_recon: float, prior=CEILING_HEIGHT_PRIOR) -> ScaleCue:
    return ScaleCue("prior_ceiling_height", "prior_ceiling", prior[0] / h_recon, prior[1] / prior[0],
                    note=f"{prior[0]}+-{prior[1]} m")


# ----------------------------------------------------------------------------------------------- fusion
def _combine(cues: list[ScaleCue]) -> tuple[float, float, dict[str, float]]:
    """Group-aware inverse-variance mean in log space. Returns (log-mean, log-sigma, weight per cue name)."""
    groups: dict[str, list[ScaleCue]] = {}
    for i, c in enumerate(cues):
        groups.setdefault(c.group or f"__{i}", []).append(c)
    gm, gv, members = [], [], []
    for g in groups.values():
        wr = np.array([1 / max(c.sigma_rand, 1e-6) ** 2 for c in g])
        x = np.array([np.log(c.scale) for c in g])
        m = float(np.sum(wr * x) / wr.sum())
        sys = float(np.sqrt(np.mean([c.sigma_sys ** 2 for c in g])))
        gm.append(m)
        gv.append(1 / wr.sum() + sys ** 2)
        members.append((g, wr / wr.sum()))
    W = 1 / np.array(gv)
    mu = float(np.sum(W * np.array(gm)) / W.sum())
    weights = {}
    for (g, wf), Wg in zip(members, W / W.sum()):
        for c, f in zip(g, wf):
            weights[c.name] = weights.get(c.name, 0.0) + float(Wg * f)
    chi2 = float(np.sum(W * (np.array(gm) - mu) ** 2))
    return mu, float(np.sqrt(1 / W.sum())), dict(weights=weights, chi2=chi2, dof=len(gm) - 1)


def fuse_scale(cues: list[ScaleCue], z_reject: float = 3.0) -> tuple[float, float, dict]:
    """Fuse scale cues. Returns (scale, sigma (absolute, 1-sigma), report).

    report: per-cue used/rejected, z-score vs. the others, weight; the 95% interval; the Birge inflation applied."""
    cues = [c for c in cues if c.scale > 0 and np.isfinite(c.scale) and np.isfinite(c.sigma) and c.sigma > 0]
    if not cues:
        return float("nan"), float("nan"), dict(cues=[], status="no scale cue")
    active = list(range(len(cues)))
    z_of: dict[int, float] = {}
    rejected: list[int] = []
    while len(active) >= 3:
        worst, zmax = None, 0.0
        for i in active:
            rest = [cues[j] for j in active if j != i]
            m, s, _ = _combine(rest)
            z = abs(np.log(cues[i].scale) - m) / np.hypot(cues[i].sigma, s)
            z_of[i] = float(z)
            if z > zmax:
                worst, zmax = i, z
        if zmax <= z_reject:
            break
        active.remove(worst)
        rejected.append(worst)
    mu, sig, info = _combine([cues[i] for i in active])
    birge = float(np.sqrt(info["chi2"] / info["dof"])) if info["dof"] > 0 else 1.0
    inflate = max(birge, 1.0)
    sig *= inflate
    scale = float(np.exp(mu))
    rep_cues = []
    for i, c in enumerate(cues):
        rep_cues.append(dict(name=c.name, kind=c.kind, scale=c.scale, rel_sigma=c.sigma, sigma_rand=c.sigma_rand,
                             sigma_sys=c.sigma_sys, group=c.group, used=i in active,
                             weight=info["weights"].get(c.name, 0.0) if i in active else 0.0,
                             z_vs_others=z_of.get(i), note=c.note))
    kinds = sorted({cues[i].kind for i in active})
    report = dict(cues=rep_cues, used_kinds=kinds, n_used=len(active), n_rejected=len(rejected),
                  chi2=info["chi2"], dof=info["dof"], birge_inflation=inflate, rel_sigma=sig,
                  ci95=(float(np.exp(mu - 1.96 * sig)), float(np.exp(mu + 1.96 * sig))),
                  status="ok" if any(k in ("sheet", "learned_depth") for k in kinds) else "priors_only")
    return scale, scale * sig, report
