"""
Rao-Blackwellized Particle Filter state for Layer 2.

Factorization
    p(x_{1:t}, {o}_i | z_{1:t}) = p(x_{1:t} | z_{1:t}) · Π_i p(o_i | x_{1:t}, z_{1:t})

* p(x_{1:t} | z_{1:t}) — approximated by weighted particles.
* p(o_i | x_{1:t}, z_{1:t}) — approximated per particle by a Gaussian on SE(3)
  (an EKF). Each particle therefore carries its own private EKF per object.

Object beliefs are stored in WORLD frame. Conditioning on a particle's
trajectory sample means the base pose has zero conditional uncertainty for
that particle, so measurements can be fused cleanly in world frame without
injecting Σ_wb into R (which would double-count). This is what gives vision
its "dual role" — the same likelihood enters both:
  1. the per-particle object EKF update
  2. the per-particle weight increment

Module layout:
    ParticleObjectBelief   — per-particle per-object (μ, Σ) in SE(3) world frame
    Particle               — T_wb + log-weight + {oid: ParticleObjectBelief}
    RBPFState              — N particles; predict / update / resample / collapse
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from pose_update.ekf_se3 import (
    se3_exp, se3_log, se3_adjoint, ekf_predict, saturate_covariance,
)
from pose_update.slam_interface import (
    PoseEstimate, ParticlePose,
    sample_particles_from_gaussian, as_gaussian,
)


# ─────────────────────────────────────────────────────────────────────
# Per-particle per-object belief
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ParticleObjectBelief:
    """Gaussian SE(3) belief about one object, conditional on one particle's
    trajectory. Stored in world frame.

    Attributes:
        mu:  (4, 4) SE(3) mean (object-in-world).
        cov: (6, 6) covariance in se(3) tangent at mu, [v, ω] ordering.
    """
    mu: np.ndarray
    cov: np.ndarray


@dataclass
class Particle:
    """One hypothesis in the RBPF.

    Attributes:
        T_wb:       (4, 4) base-to-world pose sample at the current time.
        prev_T_wb:  (4, 4) base-to-world sample at the previous step, or None
                    on the first step. Needed by the rigid-attachment predict,
                    which expresses world-frame motion of a gripper-attached
                    object as
                        ΔT_grip_w = T_wb(t) · ΔT_bg · inv(T_wb(t-1))
                    rather than the simpler similarity transform — otherwise
                    base motion with a stationary gripper would produce zero
                    object motion.
        log_weight: unnormalized log-weight (incremental likelihoods sum here).
        objects:    per-object EKF state, keyed by object id.
    """
    T_wb: np.ndarray
    log_weight: float
    objects: Dict[int, ParticleObjectBelief] = field(default_factory=dict)
    prev_T_wb: Optional[np.ndarray] = None


# ─────────────────────────────────────────────────────────────────────
# RBPF state
# ─────────────────────────────────────────────────────────────────────

class RBPFState:
    """Holds N particles and operations on them (predict / update / resample).

    The state does NOT know about object labels, manipulation phases, or
    frame counters — those are orchestrator concerns. This class is the
    pure particle-level engine.
    """

    # Numerical floor for a log-weight (avoids -inf / NaN).
    _LOG_EPS = -1e18

    def __init__(self, n_particles: int, rng: Optional[np.random.Generator] = None):
        assert n_particles >= 1
        self.n_particles = n_particles
        self.particles: List[Particle] = []
        self._rng = rng if rng is not None else np.random.default_rng()

    # --------------------------------------------------------------- #
    #  Convenience / inspection
    # --------------------------------------------------------------- #

    @property
    def initialized(self) -> bool:
        return len(self.particles) == self.n_particles

    def normalized_weights(self) -> np.ndarray:
        """Return the normalized probability vector over particles."""
        logs = np.array([p.log_weight for p in self.particles])
        m = logs.max()
        if not np.isfinite(m):
            # All -inf: degenerate; return uniform.
            return np.full(self.n_particles, 1.0 / self.n_particles)
        ws = np.exp(logs - m)
        s = ws.sum()
        if s <= 0 or not np.isfinite(s):
            return np.full(self.n_particles, 1.0 / self.n_particles)
        return ws / s

    def ess(self) -> float:
        """Effective sample size 1 / Σ w_k²."""
        ws = self.normalized_weights()
        return float(1.0 / np.sum(ws ** 2))

    # --------------------------------------------------------------- #
    #  Particle plumbing from a SLAM backend result
    # --------------------------------------------------------------- #

    def ingest_slam(self, slam_result) -> None:
        """Update per-particle T_wb from the latest SLAM result.

        * First call: creates N particles from the SLAM output.
        * Subsequent calls: overwrites each particle's T_wb in place, keeping
          object beliefs and log-weights attached to the particle slot.

        A backend that returns a `ParticlePose` with the same count transfers
        per-particle weight increments into our log-weights (additive in log
        space). Mismatched counts: we resample to match.
        """
        if isinstance(slam_result, ParticlePose):
            pp = slam_result
            if pp.n == self.n_particles:
                new_Ts = pp.particles
                new_log_increments = np.log(
                    np.clip(pp.weights, 1e-300, None))
                new_log_increments -= new_log_increments.max()
            else:
                idx = self._rng.choice(
                    pp.n, size=self.n_particles, p=pp.weights)
                new_Ts = pp.particles[idx]
                new_log_increments = np.zeros(self.n_particles)
        else:
            pe = as_gaussian(slam_result)
            sampled = sample_particles_from_gaussian(
                pe, self.n_particles, rng=self._rng)
            new_Ts = sampled.particles
            new_log_increments = np.zeros(self.n_particles)

        if not self.initialized:
            self.particles = [
                Particle(
                    T_wb=new_Ts[k].copy(),
                    log_weight=float(new_log_increments[k]),
                    objects={},
                    prev_T_wb=None,
                )
                for k in range(self.n_particles)
            ]
            return

        for k, p in enumerate(self.particles):
            # Cache previous T_wb for the rigid-attachment predict before
            # overwriting. Across resamples slot-k continuity is preserved
            # (resample deep-copies whole particles including prev_T_wb).
            p.prev_T_wb = p.T_wb
            p.T_wb = new_Ts[k].copy()
            p.log_weight += float(new_log_increments[k])

    # --------------------------------------------------------------- #
    #  Object initialization
    # --------------------------------------------------------------- #

    def ensure_object(self,
                      oid: int,
                      T_co_meas: np.ndarray,
                      init_cov: np.ndarray) -> bool:
        """Create an entry for `oid` in each particle that lacks one.

        The world-frame init is per-particle: μ^k_o = T_wb^k · T_co_meas.
        Particles disagree — that is the point; the mixture will reflect the
        base-pose uncertainty.

        Returns True if a new object entry was added in at least one particle
        (i.e., this is a newly-seen object), else False.
        """
        newly_added = False
        for p in self.particles:
            if oid in p.objects:
                continue
            T_wo = p.T_wb @ T_co_meas
            p.objects[oid] = ParticleObjectBelief(
                mu=T_wo.copy(),
                cov=init_cov.copy(),
            )
            newly_added = True
        return newly_added

    # --------------------------------------------------------------- #
    #  Predict
    # --------------------------------------------------------------- #

    def predict_objects(self, Q_fn: Callable[[int, Particle], np.ndarray],
                        P_max: Optional[np.ndarray] = None) -> None:
        """Apply per-particle EKF predict to every tracked object.

        Q_fn(oid, particle) → (6,6) process noise. This lets the caller pick
        Q based on manipulation phase, frames-since-obs, etc.

        P_max is the optional covariance-saturation cap of ekf_se3.ekf_predict
        (bernoulli_ekf.tex eq. eq:phi). Default None = no cap, matching
        pre-Bernoulli behaviour.

        Note: constant-velocity mean (T unchanged); covariance inflates by Q.
        For manipulation-set members the caller should apply
        `rigid_attachment_predict` AFTER this generic predict (the Q here is
        additive).
        """
        for p in self.particles:
            for oid, belief in p.objects.items():
                Q = Q_fn(oid, p)
                belief.mu, belief.cov = ekf_predict(
                    belief.mu, belief.cov, Q, P_max=P_max)

    def rigid_attachment_predict(self,
                                 oid: int,
                                 delta_T_grip_b: np.ndarray,
                                 Q_manip: np.ndarray) -> None:
        """Apply a rigid-attachment predict to `oid` in every particle.

        `delta_T_grip_b = T_bg(t) · inv(T_bg(t-1))` is the gripper change in
        the BASE frame (particle-independent — from proprioception).

        World-frame kinematics of an object rigidly attached to the gripper:
            T_wo(t) = T_wb(t) · T_bg(t) · T_go
            T_wo(t-1) = T_wb(t-1) · T_bg(t-1) · T_go
            ⇒ T_wo(t) = T_wb(t) · ΔT_bg · inv(T_wb(t-1)) · T_wo(t-1)

        So the world-frame transform we apply is
            ΔT_grip_w^k = T_wb^k(t) · ΔT_bg · inv(T_wb^k(t-1))

        Crucially, this formula captures both the gripper change AND the
        base change within the same transform — if the base moves but
        gripper-in-base does not, the object still moves in world.

        On the very first frame (no prev_T_wb) we fall back to the
        similarity form T_wb(t) · ΔT_bg · inv(T_wb(t)), which is only
        correct when the base is stationary.

        Covariance: μ ← ΔT_grip_w · μ; Σ ← Ad(ΔT_grip_w) · Σ · Ad(.)ᵀ + Q_manip.
        """
        for p in self.particles:
            belief = p.objects.get(oid)
            if belief is None:
                continue
            T_wb_prev = p.prev_T_wb if p.prev_T_wb is not None else p.T_wb
            delta_T_grip_w = p.T_wb @ delta_T_grip_b @ np.linalg.inv(T_wb_prev)
            Ad = se3_adjoint(delta_T_grip_w)
            belief.mu = delta_T_grip_w @ belief.mu
            belief.cov = Ad @ belief.cov @ Ad.T + Q_manip
            belief.cov = 0.5 * (belief.cov + belief.cov.T)

    # --------------------------------------------------------------- #
    #  Measurement update + likelihood (vision's dual role)
    # --------------------------------------------------------------- #

    def update_observation(self,
                           oid: int,
                           T_co_meas: np.ndarray,
                           R_icp: np.ndarray,
                           iekf_iters: int = 2,
                           huber_w: float = 1.0,
                           P_max: Optional[np.ndarray] = None) -> None:
        """Per-particle IEKF update + per-particle likelihood accumulation.

        For particle k:
            T_wo_meas^k = T_wb^k · T_co_meas
            δ^k         = log( μ^k · exp(-K δ) ... ) via IEKF relinearization
            S^k         = Σ^k_o + R_icp / w
            log p(z|x^k, o^k) = -½ δ^kᵀ S^{-1} δ^k - ½ log det(2πS)

        The log-likelihood is added to particle.log_weight — that is how
        vision reweights the trajectory posterior (in addition to updating
        the object EKF). This is the whole point of RBPF here.

        R_icp is treated as a world-frame 6×6 covariance; callers who have
        noise in camera frame should transform with Ad(T_wb · T_bc) upstream.

        Args:
            huber_w: Huber redescending M-estimator weight in [0, 1] from
                ekf_se3.huber_weight(). Scales R by 1/w so the gain shrinks
                with d^2. Default 1.0 = no Huber (pre-Bernoulli behaviour).
                A caller passing w = 0 should NOT invoke this method (route
                the detection to the missed branch instead); we treat it as
                a clamp to a large reweight for numerical safety only.
            P_max: optional (6, 6) covariance-saturation cap applied to the
                posterior (bernoulli_ekf.tex eq. eq:phi). Default None =
                no cap, matching pre-Bernoulli behaviour.
        """
        R_sym = 0.5 * (R_icp + R_icp.T)
        if huber_w > 0.0 and huber_w < 1.0:
            R_sym = R_sym / huber_w
        I6 = np.eye(6)
        two_pi_log = 6.0 * np.log(2.0 * np.pi)

        for p in self.particles:
            belief = p.objects.get(oid)
            if belief is None:
                continue

            # S and K are constant during IEKF (they depend only on the
            # prior covariance, which is not updated until after the loop).
            S = belief.cov + R_sym
            K = np.linalg.solve(S.T, belief.cov.T).T  # Σ · S⁻¹

            T_wo_meas = p.T_wb @ T_co_meas
            mu_lin = belief.mu.copy()
            delta0 = None  # innovation at the prior mean — for likelihood

            for i in range(max(1, iekf_iters)):
                delta = se3_log(np.linalg.inv(mu_lin) @ T_wo_meas)
                if i == 0:
                    delta0 = delta
                mu_lin = belief.mu @ se3_exp(K @ delta)

            # Likelihood at the prior-mean innovation (standard Kalman form).
            sign, logdet = np.linalg.slogdet(S)
            if sign <= 0 or not np.isfinite(logdet):
                log_lik = self._LOG_EPS
            else:
                try:
                    Sinv_delta = np.linalg.solve(S, delta0)
                    log_lik = (-0.5 * float(delta0 @ Sinv_delta)
                               - 0.5 * (float(logdet) + two_pi_log))
                except np.linalg.LinAlgError:
                    log_lik = self._LOG_EPS

            # Joseph-form covariance update (bernoulli_ekf.tex eq. eq:ekf_P_upd):
            #     P_{k|k} = (I - K) P_{k|k-1} (I - K)^T + K R K^T
            # Equivalent to (I-K)P under exact arithmetic, but PSD-preserving
            # under rounding error when K drifts from optimal (e.g., with
            # Huber re-weighting R <- R/w). We then optionally apply Phi.
            I_K = I6 - K
            cov_post = I_K @ belief.cov @ I_K.T + K @ R_sym @ K.T
            cov_post = 0.5 * (cov_post + cov_post.T)
            if P_max is not None:
                cov_post = saturate_covariance(cov_post, P_max)
            belief.mu = mu_lin
            belief.cov = cov_post

            p.log_weight += log_lik

    # --------------------------------------------------------------- #
    #  Innovation statistics for data association
    # --------------------------------------------------------------- #

    def innovation_stats(self,
                         oid: int,
                         T_co_meas: np.ndarray,
                         R_icp: np.ndarray) -> Optional[tuple]:
        """Weighted-particle-averaged innovation quantities for a
        (track, measurement) pair; used to build the Hungarian cost matrix.

        Returns (nu, S, d2, log_lik) where:
            nu    = innovation in se(3) tangent at the collapsed prior mean
            S     = residual covariance (Sigma_obj + R_icp), collapsed
            d2    = nu^T S^{-1} nu
            log_lik = Gaussian log-likelihood of the innovation (cf.
                      bernoulli_ekf.tex eq. eq:ekf_lik)

        None is returned if `oid` does not exist in any particle.
        """
        collapsed = self.collapsed_object(oid)
        if collapsed is None:
            return None
        T_prior = collapsed.T
        cov_prior = collapsed.cov

        # The measurement T_co is camera-frame; we project it to world frame
        # using the collapsed base pose (mean of the particle cloud). This
        # is the per-formulation single-Gaussian approximation used in data
        # association; per-particle updates remain mixture-exact.
        base_pe = self.collapsed_base()
        T_wo_meas = base_pe.T @ T_co_meas

        nu = se3_log(np.linalg.inv(T_prior) @ T_wo_meas)
        R_sym = 0.5 * (R_icp + R_icp.T)
        S = cov_prior + R_sym
        S = 0.5 * (S + S.T)

        sign, logdet = np.linalg.slogdet(S)
        try:
            Sinv_nu = np.linalg.solve(S, nu)
            d2 = float(nu @ Sinv_nu)
        except np.linalg.LinAlgError:
            d2 = float("inf")
            Sinv_nu = None
        if sign <= 0 or not np.isfinite(logdet) or not np.isfinite(d2):
            log_lik = self._LOG_EPS
        else:
            two_pi_log = 6.0 * np.log(2.0 * np.pi)
            log_lik = -0.5 * d2 - 0.5 * (float(logdet) + two_pi_log)
        return nu, S, d2, log_lik

    # --------------------------------------------------------------- #
    #  Object deletion
    # --------------------------------------------------------------- #

    def delete_object(self, oid: int) -> bool:
        """Remove `oid` from every particle. Returns True if any particle
        held this object, else False."""
        removed = False
        for p in self.particles:
            if oid in p.objects:
                del p.objects[oid]
                removed = True
        return removed

    # --------------------------------------------------------------- #
    #  Resampling
    # --------------------------------------------------------------- #

    def resample_if_needed(self, threshold_frac: float = 0.5) -> bool:
        """Systematic resample if ESS drops below `threshold_frac · N`.

        On resample: each selected particle is deep-copied (T_wb + per-object
        (μ, Σ)) and its log-weight reset to 0 (uniform in log-space).

        Returns True if resampling fired, else False.
        """
        N = self.n_particles
        if self.ess() >= threshold_frac * N:
            return False

        ws = self.normalized_weights()
        cumsum = np.cumsum(ws)
        # Guard against float creep
        cumsum[-1] = 1.0
        u0 = float(self._rng.uniform(0.0, 1.0 / N))
        u = u0 + np.arange(N) / N
        idx = np.searchsorted(cumsum, u)
        idx = np.clip(idx, 0, N - 1)

        new_particles = []
        for i in idx:
            src = self.particles[int(i)]
            clone = Particle(
                T_wb=src.T_wb.copy(),
                log_weight=0.0,
                objects={
                    oid: ParticleObjectBelief(
                        mu=b.mu.copy(), cov=b.cov.copy())
                    for oid, b in src.objects.items()
                },
                prev_T_wb=(src.prev_T_wb.copy()
                           if src.prev_T_wb is not None else None),
            )
            new_particles.append(clone)
        self.particles = new_particles
        return True

    # --------------------------------------------------------------- #
    #  Collapsing to single-Gaussian summaries (for legacy consumers)
    # --------------------------------------------------------------- #

    def collapsed_base(self) -> PoseEstimate:
        """Moment-match the base-pose particles to a single SE(3) Gaussian."""
        ws = self.normalized_weights()
        Ts = np.stack([p.T_wb for p in self.particles], axis=0)
        return ParticlePose(particles=Ts, weights=ws).to_gaussian()

    def collapsed_object(self, oid: int) -> Optional[PoseEstimate]:
        """Mixture-of-Gaussians → single Gaussian summary for one object.

        Uses the standard formula:
            μ̄  = Lie-group weighted mean of {μ_k}
            Σ̄  = Σ_k w_k · Σ_k  +  Σ_k w_k · (μ_k ⊖ μ̄)(μ_k ⊖ μ̄)ᵀ

        `ParticlePose.to_gaussian` already computes the mean AND the
        second (spread) term, so we just add the expected per-particle
        cov `E[Σ_k]` on top.

        Returns None if no particle has this object yet.
        """
        ws_all = self.normalized_weights()
        mus: List[np.ndarray] = []
        covs: List[np.ndarray] = []
        ws: List[float] = []
        for p, w in zip(self.particles, ws_all):
            b = p.objects.get(oid)
            if b is None:
                continue
            mus.append(b.mu)
            covs.append(b.cov)
            ws.append(w)
        if not mus:
            return None
        w_sum = float(np.sum(ws))
        if w_sum <= 0:
            return None
        ws_arr = np.asarray(ws) / w_sum

        pp_obj = ParticlePose(
            particles=np.stack(mus, axis=0),
            weights=ws_arr,
        )
        mean_pe = pp_obj.to_gaussian()  # .cov is the spread term

        expected_cov = sum(w * c for w, c in zip(ws_arr, covs))
        cov_total = mean_pe.cov + expected_cov
        cov_total = 0.5 * (cov_total + cov_total.T)
        return PoseEstimate(T=mean_pe.T, cov=cov_total)

    def collapsed_objects(self) -> Dict[int, PoseEstimate]:
        """Collapse every tracked object to a single-Gaussian summary."""
        oids = set()
        for p in self.particles:
            oids.update(p.objects.keys())
        out: Dict[int, PoseEstimate] = {}
        for oid in oids:
            pe = self.collapsed_object(oid)
            if pe is not None:
                out[oid] = pe
        return out

    # --------------------------------------------------------------- #
    #  Slow-tier reconcile (Option A: shift-and-inject collapsed posterior)
    # --------------------------------------------------------------- #

    def inject_posterior(self, oid: int, posterior: PoseEstimate) -> None:
        """Shift every particle's belief for `oid` toward a single posterior.

        Each particle's mean is set to the posterior mean; each particle's
        covariance is set to the posterior covariance. This is Option A from
        the plan — loses mixture structure, but is the minimal change that
        lets the slow-tier re-ingested raw observations flow back into the
        fast tier.
        """
        for p in self.particles:
            if oid in p.objects:
                p.objects[oid] = ParticleObjectBelief(
                    mu=posterior.T.copy(),
                    cov=posterior.cov.copy(),
                )
