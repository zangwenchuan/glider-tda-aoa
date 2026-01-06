import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass

try:
    from glider_params_private import GLIDER_PARAMS  # type: ignore
except Exception:
    GLIDER_PARAMS = {
        "mass": {"mh": 30.0, "mm": 2.0, "mb": 2.287},
        "geometry": {"rh1": -0.65, "rh3": 0.12, "rb1": -0.45, "rbattary": 0.10},
        "added_mass": {"mf11": 1.2, "mf22": 1.3, "mf33": 1.5},
        "inertia_fluid": {"jf11": 0.15, "jf22": 0.18, "jf33": 0.20},
        "inertia_hull": {"jh11": 12.0, "jh22": 14.0, "jh33": 16.0},
        "hydrodynamics": {"KD0": 0.1, "KD": 0.05, "KL0": 0.2, "KL": 0.08, "Kb": 0.03},
        "moments": {"KMR": 0.01, "Kp": 0.02, "KM0": 0.015, "KM": 0.04, "Kq": 0.03, "KMY": 0.01, "Kr": 0.025},
        "physics": {"g": 9.81},
        "tda": {
            "lambda_euler": 1.0,  
            "lambda_depth": 1.0,
            "lambda_dz": 1.0
        }
    }


# =========================================================
# Sensor/actuator mapping (open demo)
# =========================================================
def pad2rm1(padi, smin=500, smid=1500, smax=2500, mmid=-0.5, mmin=-0.6, mmax=-0.4):
    srange = smax - smin
    mrange = mmax - mmin
    snormalized = (padi - smid) / (srange / 2)
    rm1 = mmid + snormalized * mrange / 2
    return float(rm1)

def oad2vb(oad, sensor_min=500, sensor_mid=1000, sensor_max=1500, model_min=0, model_mid=0.5, model_max=1.0):
    sensor_range = sensor_max - sensor_min
    model_range = model_max - model_min
    sensor_value_normalized = (oad - sensor_mid) / (sensor_range / 2)
    vb = model_mid + sensor_value_normalized * (model_range / 2)
    return float(vb)

def cal_mb(vb, oil_density=0.8, water_density=1.0, v_bmax=1.0):
    oil_mass = v_bmax * oil_density
    water_mass = (v_bmax - vb) * water_density
    return float(oil_mass + water_mass)

def rad2rot(rad, sensor_min=1000, sensor_max=2000, model_min=0, model_max=30):
    sensor_range = sensor_max - sensor_min
    model_range = model_max - model_min
    sensor_value_normalized = (rad - sensor_min) / sensor_range
    rot = model_min + sensor_value_normalized * model_range
    return float(rot)


# =========================================================
# Euler mapping
# =========================================================
def R_bn(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    Body -> NED rotation matrix R(φ,θ,ψ)

    """
    sphi, cphi = np.sin(phi), np.cos(phi)
    sth, cth = np.sin(theta), np.cos(theta)
    sps, cps = np.sin(psi), np.cos(psi)

    return np.array([
        [cps*cth, cps*sth*sphi - sps*cphi, cps*sth*cphi + sps*sphi],
        [sps*cth, sps*sth*sphi + cps*cphi, sps*sth*cphi - cps*sphi],
        [-sth,    cth*sphi,               cth*cphi]
    ], dtype=float)

def J_euler(phi: float, theta: float) -> np.ndarray:
    """
    Euler-rate mapping 

    """
    sphi, cphi = np.sin(phi), np.cos(phi)
    sth, cth = np.sin(theta), np.cos(theta)

    eps = 1e-9
    if abs(cth) < eps:
        cth = eps if cth >= 0 else -eps

    return np.array([
        [1.0, sphi*sth/cth, cphi*sth/cth],
        [0.0, cphi,         -sphi],
        [0.0, sphi/cth,     cphi/cth]
    ], dtype=float)

def R_fb(alpha: float, beta: float) -> np.ndarray:
    """
    Flow -> Body rotation R_FB(α,β)

    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)

    return np.array([
        [ca*cb, -ca*sb, -sa],
        [sb,     cb,     0.0],
        [sa*cb, -sa*sb,  ca]
    ], dtype=float)


# =========================================================
# State and Model
# =========================================================
@dataclass
class GliderState:
    # η = [x,y,z,phi,theta,psi] (NED + Euler)
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    phi: float = 0.0
    theta: float = 0.0
    psi: float = 0.0
    # ν = [u,v,w,p,q,r] (body)
    u: float = 0.0
    v: float = 0.0
    w: float = 0.0
    p: float = 0.0
    q: float = 0.0
    r: float = 0.0


class GliderModel:

    def __init__(self, state: GliderState | None = None):
        self.s = state if state is not None else GliderState()

    # ----------------------------
    # Hydrodynamic forces/moments
    # ----------------------------
    def hydrodynamics(self, alpha: float, beta: float) -> tuple[float, float, float, float, float, float]:
        KD0 = GLIDER_PARAMS["hydrodynamics"]["KD0"]
        KD  = GLIDER_PARAMS["hydrodynamics"]["KD"]
        KL0 = GLIDER_PARAMS["hydrodynamics"]["KL0"]
        KL  = GLIDER_PARAMS["hydrodynamics"]["KL"]
        Kb  = GLIDER_PARAMS["hydrodynamics"]["Kb"]

        KMR = GLIDER_PARAMS["moments"]["KMR"]
        Kp  = GLIDER_PARAMS["moments"]["Kp"]
        KM0 = GLIDER_PARAMS["moments"]["KM0"]
        KM  = GLIDER_PARAMS["moments"]["KM"]
        Kq  = GLIDER_PARAMS["moments"]["Kq"]
        KMY = GLIDER_PARAMS["moments"]["KMY"]
        Kr  = GLIDER_PARAMS["moments"]["Kr"]

        u, v, w = self.s.u, self.s.v, self.s.w
        V = float(np.sqrt(u*u + v*v + w*w)) + 1e-12

        D  = (KD0 + KD * alpha*alpha) * (V**2)
        L  = (KL0 + KL * alpha)       * (V**2)
        SF = (Kb * beta)              * (V**2)

        # Parametric moments (matches your coefficient form)
        p, q, r = self.s.p, self.s.q, self.s.r
        Mx = (KMR * beta + Kp * p)        * (V**2)
        My = (KM0 + KM * alpha + Kq * q)  * (V**2)
        Mz = (KMY * beta + Kr * r)        * (V**2)

        return D, SF, L, Mx, My, Mz

    # ----------------------------
    # Total force in body frame
    # ----------------------------
    def total_force_body(self, m0: float, alpha: float, beta: float) -> np.ndarray:
        g = GLIDER_PARAMS["physics"]["g"]
        phi, theta = self.s.phi, self.s.theta

        # buoyancy/gravity 
        g_vec = np.array([
            np.sin(theta),
            -np.sin(phi) * np.cos(theta),
            -np.cos(phi) * np.cos(theta)
        ], dtype=float) * (m0 * g)

        # hydrodynamic force 
        D, SF, L, _, _, _ = self.hydrodynamics(alpha, beta)
        F_flow = np.array([-D, SF, -L], dtype=float)
        F_body = R_fb(alpha, beta) @ F_flow

        return g_vec + F_body

    # ----------------------------
    # Total moment in body frame
    # ----------------------------
    def total_moment_body(self, r_cm: np.ndarray, alpha: float, beta: float, mb_current: float) -> np.ndarray:
        g = GLIDER_PARAMS["physics"]["g"]

        mh = GLIDER_PARAMS["mass"]["mh"]
        mm = GLIDER_PARAMS["mass"]["mm"]
        m = mh + mm + mb_current

        phi, theta = self.s.phi, self.s.theta
        g_body = np.array([
            np.sin(theta),
            -np.sin(phi) * np.cos(theta),
            -np.cos(phi) * np.cos(theta)
        ], dtype=float) * (m * g)

        tau_g = np.cross(r_cm, g_body)

        _, _, _, Mx, My, Mz = self.hydrodynamics(alpha, beta)
        tau_h = np.array([Mx, My, Mz], dtype=float)

        return tau_g + tau_h


    def step(self,
             dt: float,
             *,
             m0: float,
             r_cm: np.ndarray,
             alpha: float,
             mb_current: float | None = None,
             override_euler: tuple[float, float, float] | None = None,
             override_depth: float | None = None,
             override_dz: float | None = None,
             override_body_rates: tuple[float, float, float] | None = None) -> GliderState:

        # --- current mass ---
        mh = GLIDER_PARAMS["mass"]["mh"]
        mm = GLIDER_PARAMS["mass"]["mm"]
        mb0 = GLIDER_PARAMS["mass"]["mb"]
        mb = float(mb_current) if mb_current is not None else float(mb0)
        m = mh + mm + mb

        # --- added mass and inertia ---
        mf11 = GLIDER_PARAMS["added_mass"]["mf11"]
        mf22 = GLIDER_PARAMS["added_mass"]["mf22"]
        mf33 = GLIDER_PARAMS["added_mass"]["mf33"]

        jh11 = GLIDER_PARAMS["inertia_hull"]["jh11"]
        jh22 = GLIDER_PARAMS["inertia_hull"]["jh22"]
        jh33 = GLIDER_PARAMS["inertia_hull"]["jh33"]

        jf11 = GLIDER_PARAMS["inertia_fluid"]["jf11"]
        jf22 = GLIDER_PARAMS["inertia_fluid"]["jf22"]
        jf33 = GLIDER_PARAMS["inertia_fluid"]["jf33"]

        # translational effective mass (diagonal)
        Mv = np.diag([m + mf11, m + mf22, m + mf33])
        # rotational inertia (diagonal)
        J = np.diag([jh11 + jf11, jh22 + jf22, jh33 + jf33])

        v_b = np.array([self.s.u, self.s.v, self.s.w], dtype=float)
        w_b = np.array([self.s.p, self.s.q, self.s.r], dtype=float)
        beta = np.arcsin(self.s.v/np.sqrt(self.s.u**2+self.s.v**2+self.s.w**2)) 

        F_b = self.total_force_body(m0=m0, alpha=alpha, beta=beta)
        tau_b = self.total_moment_body(r_cm=r_cm, alpha=alpha, beta=beta, mb_current=mb)

        #  v_dot (simplified rigid-body style with diagonal Mv) ---
        v_dot = np.linalg.solve(Mv, (F_b - np.cross(w_b, (Mv @ v_b))))

        #   omega_dot ---
        w_dot = np.linalg.solve(J, (tau_b - np.cross(w_b, (J @ w_b))))

        # integrate ν
        v_b = v_b + v_dot * dt
        w_b = w_b + w_dot * dt

        # kinematics ---
        R = R_bn(self.s.phi, self.s.theta, self.s.psi)
        pos_dot = R @ v_b                   
        euler_dot = J_euler(self.s.phi, self.s.theta) @ w_b

        # =========================================================
        # TDA: assimilation BEFORE integrating z (use override_dz)
        # =========================================================
        tda = GLIDER_PARAMS.get("tda", {})
        lam_euler = float(tda.get("lambda_euler", 1.0))
        lam_depth = float(tda.get("lambda_depth", 1.0))
        lam_dz = float(tda.get("lambda_dz", 1.0))

        # 1) Optional:  assimilate vertical velocity z_dot (NED)
        if override_dz is not None:
            z_dot_obs = float(override_dz)
            pos_dot[2] = (1.0 - lam_dz) * float(pos_dot[2]) + lam_dz * z_dot_obs

        # integrate η (position)
        self.s.x += float(pos_dot[0] * dt)
        self.s.y += float(pos_dot[1] * dt)

        z_pred = float(self.s.z + pos_dot[2] * dt)
        self.s.z = z_pred

        # integrate Euler from predicted euler_dot
        self.s.phi   += float(euler_dot[0] * dt)
        self.s.theta += float(euler_dot[1] * dt)
        self.s.psi   += float(euler_dot[2] * dt)

        # write back ν
        self.s.u, self.s.v, self.s.w = float(v_b[0]), float(v_b[1]), float(v_b[2])
        self.s.p, self.s.q, self.s.r = float(w_b[0]), float(w_b[1]), float(w_b[2])

        # 2) Optional: assimilate Euler angles (sequential relaxation)
        if override_euler is not None:
            phi_obs, theta_obs, psi_obs = override_euler
            self.s.phi   = (1.0 - lam_euler) * self.s.phi   + lam_euler * float(phi_obs)
            self.s.theta = (1.0 - lam_euler) * self.s.theta + lam_euler * float(theta_obs)
            self.s.psi   = (1.0 - lam_euler) * self.s.psi   + lam_euler * float(psi_obs)

        # 3) Optional: assimilate depth directly (sequential relaxation)
        if override_depth is not None:
            self.s.z = (1.0 - lam_depth) * self.s.z + lam_depth * float(override_depth)

        # 4) Optional: assimilate body rates
        if override_body_rates is not None:
            p_obs, q_obs, r_obs = override_body_rates
            self.s.p, self.s.q, self.s.r = float(p_obs), float(q_obs), float(r_obs)

        return self.s


# =========================================================
# Main: runnable script for open-source repository
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Open-source TDA support code (equation-aligned).")
    parser.add_argument("--input", type=str, required=True,
                        help="CSV file with columns: PAD,RAD,OAD,aoa,roll,pitch,heading,depth,dz")
    parser.add_argument("--out", type=str, default="outputs/trajectory.csv")
    parser.add_argument("--steps", type=int, default=10, help="Substeps per sample interval")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    # Required columns
    required = ["PAD", "RAD", "OAD", "aoa", "roll", "pitch", "heading", "depth", "dz"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in input CSV.")

    PAD = df["PAD"].to_numpy()
    RAD = df["RAD"].to_numpy()
    OAD = df["OAD"].to_numpy()
    aoa = df["aoa"].to_numpy()
    roll = df["roll"].to_numpy()
    pitch = df["pitch"].to_numpy()
    heading = df["heading"].to_numpy()
    dep = df["depth"].to_numpy()
    dz = df["dz"].to_numpy()

    steps = int(args.steps)
    dt_sub = 1.0 / steps   

    rbattary = GLIDER_PARAMS["geometry"]["rbattary"]

    model = GliderModel()

    # buoyancy baseline
    mb0 = None

    rows = []
    N = len(dep)

    for i in range(N * steps):
        k = i // steps

        rm1 = pad2rm1(PAD[k])
        rot = rad2rot(RAD[k])
        vb = oad2vb(OAD[k])
        mb = cal_mb(vb)

        if mb0 is None:
            mb0 = mb
        m0 = mb - mb0  # net buoyant term

        # r_cm (movable mass vector in body)
        r2 = rbattary * np.sin(rot * np.pi / 180.0)
        r3 = rbattary * np.cos(rot * np.pi / 180.0)
        r_cm = np.array([rm1, r2, r3], dtype=float)

        alpha = float(aoa[k]) * np.pi / 180.0

        override_euler = (
            float(roll[k]) * np.pi / 180.0,
            float(pitch[k]) * np.pi / 180.0,
            float(heading[k]) * np.pi / 180.0
        )

        override_depth = float(dep[k])
        override_dz = float(dz[k])

        s = model.step(
            dt=dt_sub,
            m0=m0,
            r_cm=r_cm,
            alpha=alpha,
            mb_current=mb,
            override_euler=override_euler,
            override_depth=override_depth,
            override_dz=override_dz
        )
                # record once per sample (end of each macro step)
        if i % steps == 0:
            rows.append({
                "k": k,
                "x": s.x, "y": s.y, "z": s.z,
                "phi": s.phi, "theta": s.theta, "psi": s.psi,
                "u": s.u, "v": s.v, "w": s.w,
                "p": s.p, "q": s.q, "r": s.r
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"[OK] Saved: {args.out}")

 

if __name__ == "__main__":
    main()
