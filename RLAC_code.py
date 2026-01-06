"""
AOA inference training script.

- Implements a DDPG with TF2/Keras.
- Environment uses measured depth/heading/pitch series to infer horizontal drift
  and optimizes a correction alpha (AOA bias) to reach a target (xt, yt).

Usage:
  1) Local data:
     python train_aoa_actor_critic.py --data G_sample.xlsx --out_dir outputs

  2) Demo (synthetic) data:
     python train_aoa_actor_critic.py --demo --out_dir outputs
"""

import argparse
import os
import random
from dataclasses import dataclass
from math import pi, tan, sin, cos, sqrt

import numpy as np
import pandas as pd
import tensorflow as tf


# =========================
# Reproducibility
# =========================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# =========================
# Hyperparameters container
# =========================
@dataclass
class TrainConfig:
    batch_size: int = 128
    lr_actor: float = 1.3e-3
    lr_critic: float = 1.3e-3
    gamma: float = 0.99
    tau: float = 0.02
    episodes: int = 501
    memory_capacity: int = 50000
    epsilon: float = 1.0
    epsilon_decay: float = 0.6
    epsilon_min: float = 0.1
    action_scale: float = 10.8
    # target
    xt: float = 100.0
    yt: float = -16.0
    goal_tole: float = 10.0
    # env heuristics
    pathlow: float = 0.5          # threshold for small pitch+alpha (deg)
    convert_factor: float = 1.5   # fallback scaling when near 0
    sigma: float = 10.0           # reward sigma
    phase: int = 110              # phase split index (int)



# =========================
# Actor / Critic
# =========================
class Actor(tf.keras.Model):
    def __init__(self, n_actions: int, action_scale: float):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(256, activation="relu")
        self.fc2 = tf.keras.layers.Dense(256, activation="relu")
        self.out = tf.keras.layers.Dense(n_actions, activation="tanh")
        self.action_scale = float(action_scale)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        a = self.out(x) * self.action_scale
        return a


class Critic(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(256, activation="relu")
        self.fc2 = tf.keras.layers.Dense(256, activation="relu")
        self.q_out = tf.keras.layers.Dense(1)

    def call(self, state, action):
        action = tf.reshape(action, [-1, 1])  # (B,1) for concat
        x = tf.concat([state, action], axis=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.q_out(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.buffer = []

    def push(self, transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s0, a0, r1, s1 = zip(*batch)
        return (
            np.array(s0, dtype=np.float32),
            np.array(a0, dtype=np.float32),
            np.array(r1, dtype=np.float32).reshape(-1, 1),
            np.array(s1, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DDPGAgent:
    def __init__(self, n_states: int, n_actions: int, cfg: TrainConfig):
        self.n_actions = n_actions
        self.cfg = cfg

        self.eval_actor = Actor(n_actions=n_actions, action_scale=cfg.action_scale)
        self.target_actor = Actor(n_actions=n_actions, action_scale=cfg.action_scale)
        self.eval_critic = Critic()
        self.target_critic = Critic()

        # Build networks by calling once
        dummy_s = tf.zeros((1, n_states), dtype=tf.float32)
        dummy_a = tf.zeros((1, n_actions), dtype=tf.float32)
        _ = self.eval_actor(dummy_s)
        _ = self.target_actor(dummy_s)
        _ = self.eval_critic(dummy_s, dummy_a)
        _ = self.target_critic(dummy_s, dummy_a)

        self.target_actor.set_weights(self.eval_actor.get_weights())
        self.target_critic.set_weights(self.eval_critic.get_weights())

        self.actor_optim = tf.keras.optimizers.Adam(learning_rate=cfg.lr_actor)
        self.critic_optim = tf.keras.optimizers.Adam(learning_rate=cfg.lr_critic)

        self.buffer = ReplayBuffer(cfg.memory_capacity)

    def choose_action(self, state: np.ndarray, epsilon: float) -> np.ndarray:
        if np.random.uniform() <= epsilon:
            a = np.random.uniform(-self.cfg.action_scale, self.cfg.action_scale, self.n_actions).astype(np.float32)
        else:
            s = tf.convert_to_tensor([state], dtype=tf.float32)
            a = self.eval_actor(s).numpy().squeeze().astype(np.float32)
            a = np.array([a], dtype=np.float32) if self.n_actions == 1 else a
        return a

    def store_transition(self, s, a, r, s_):
        self.buffer.push((s, a, r, s_))

    @tf.function
    def _soft_update(self, target_vars, eval_vars, tau: float):
        for t, e in zip(target_vars, eval_vars):
            t.assign(t * (1.0 - tau) + e * tau)

    def learn(self):
        if len(self.buffer) < self.cfg.batch_size:
            return None, None

        s0, a0, r1, s1 = self.buffer.sample(self.cfg.batch_size)
        s0 = tf.convert_to_tensor(s0, dtype=tf.float32)
        a0 = tf.convert_to_tensor(a0, dtype=tf.float32)
        r1 = tf.convert_to_tensor(r1, dtype=tf.float32)
        s1 = tf.convert_to_tensor(s1, dtype=tf.float32)

        # Critic update
        with tf.GradientTape() as tape_c:
            a1 = self.target_actor(s1)
            y_true = r1 + self.cfg.gamma * self.target_critic(s1, a1)
            y_pred = self.eval_critic(s0, a0)
            critic_loss = tf.reduce_mean(tf.square(y_true - y_pred))

        c_grads = tape_c.gradient(critic_loss, self.eval_critic.trainable_variables)
        self.critic_optim.apply_gradients(zip(c_grads, self.eval_critic.trainable_variables))

        # Actor update
        with tf.GradientTape() as tape_a:
            actor_loss = -tf.reduce_mean(self.eval_critic(s0, self.eval_actor(s0)))

        a_grads = tape_a.gradient(actor_loss, self.eval_actor.trainable_variables)
        self.actor_optim.apply_gradients(zip(a_grads, self.eval_actor.trainable_variables))

        # Soft update
        self._soft_update(self.target_actor.trainable_variables, self.eval_actor.trainable_variables, self.cfg.tau)
        self._soft_update(self.target_critic.trainable_variables, self.eval_critic.trainable_variables, self.cfg.tau)

        return float(actor_loss.numpy()), float(critic_loss.numpy())
    
    def save(self, ckpt_dir: str) -> None:
        os.makedirs(ckpt_dir, exist_ok=True)
        self.eval_actor.save_weights(os.path.join(ckpt_dir, "actor.weights.h5"))
        self.eval_critic.save_weights(os.path.join(ckpt_dir, "critic.weights.h5"))

    def load(self, ckpt_dir: str, n_states: int) -> None:
        # ensure variables exist
        dummy_s = tf.zeros((1, n_states), dtype=tf.float32)
        _ = self.eval_actor(dummy_s)
        _ = self.eval_critic(dummy_s, tf.zeros((1, self.n_actions), dtype=tf.float32))

        self.eval_actor.load_weights(os.path.join(ckpt_dir, "actor.weights.h5"))
        self.eval_critic.load_weights(os.path.join(ckpt_dir, "critic.weights.h5"))

        # keep targets consistent
        self.target_actor.set_weights(self.eval_actor.get_weights())
        self.target_critic.set_weights(self.eval_critic.get_weights())



# =========================
# Environment: AOA inference on measured series
# =========================
class DriftEnv:
    """
    State: [x, y, phase]
    Action: alpha (AOA correction, degrees), clipped to [-action_scale, action_scale]
    """

    def __init__(self, depth: np.ndarray, heading: np.ndarray, pitch: np.ndarray, cfg: TrainConfig):
        self.cfg = cfg   

        self.depth = np.asarray(depth, dtype=float)
        self.heading = np.asarray(heading, dtype=float)
        self.pitch = np.asarray(pitch, dtype=float)
        self.len = int(len(self.depth))

        self.xt = float(cfg.xt)
        self.yt = float(cfg.yt)
        self.goal_tole = float(cfg.goal_tole)
        self.a_low = -float(cfg.action_scale)
        self.a_high = float(cfg.action_scale)

        self.x0 = 0.0
        self.y0 = 0.0
        self.dis0 = float(np.sqrt((self.x0 - self.xt) ** 2 + (self.y0 - self.yt) ** 2))

        self.rad_factor = pi / 180.0
        self.reset()

    def reset(self):
        self.x = self.x0
        self.y = self.y0
        self.done = False
        self.dis = self.dis0
        phase = -1.0
        return np.array([self.x, self.y, phase], dtype=np.float32)

    def step(self, i: int, alpha_deg: float):
        pitch_alpha_rad = (alpha_deg + self.pitch[i]) * self.rad_factor
        gr = self.heading[i] * self.rad_factor

        # vertical-to-horizontal estimate
        if i == 0:
            denom = tan(pitch_alpha_rad) if abs(tan(pitch_alpha_rad)) > 1e-9 else 1e-9
            vh = abs(self.depth[i] / denom)
        else:
            ddep = float(self.depth[i] - self.depth[i - 1])
            if abs(alpha_deg + self.pitch[i]) < self.cfg.pathlow:
                vh = abs(ddep) * self.cfg.convert_factor
            else:
                denom = tan(pitch_alpha_rad) if abs(tan(pitch_alpha_rad)) > 1e-9 else 1e-9
                vh = abs(ddep / denom)

        hd = float(self.heading[i])
        if 0 < hd <= 90:
            vy = vh * cos(gr)
            vx = vh * sin(gr)
        elif 90 < hd <= 180:
            vy = -vh * sin(gr - pi / 2)
            vx = vh * cos(gr - pi / 2)
        elif 180 < hd <= 270:
            vy = -vh * cos(gr - pi)
            vx = -vh * sin(gr - pi)
        else:
            vy = vh * sin(gr - 3 * pi / 2)
            vx = -vh * cos(gr - 3 * pi / 2)

        self.x += float(vx)
        self.y += float(vy)

        if i == self.len - 1:
            self.done = True

        dis = float(np.sqrt((self.x - self.xt) ** 2 + (self.y - self.yt) ** 2))

        # reward
        r = -(1.0 - np.exp(-((self.x - self.xt) ** 2 + (self.y - self.yt) ** 2) / (2.0 * (self.cfg.sigma ** 2))))

        # phase flag
        phase = -1.0 if i <= int(self.cfg.phase) else 1.0

        self.dis = dis
        s_ = np.array([self.x, self.y, phase], dtype=np.float32)
        return s_, float(r), bool(self.done)



# =========================
# Data loading
# =========================
def load_series(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Expect columns: Gdepth, Gheading, Gpitch
    """
    if path.lower().endswith(".xlsx") or path.lower().endswith(".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    for c in ["Gdepth", "Gheading", "Gpitch"]:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in data file. Required: Gdepth,Gheading,Gpitch")

    depth = df["Gdepth"].to_numpy()
    heading = df["Gheading"].to_numpy()
    pitch = df["Gpitch"].to_numpy()
    return depth, heading, pitch


def make_demo_series(n: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthetic demo data so the repo runs without private sea-trial files.
    """
    t = np.arange(n)
    depth = np.cumsum(np.maximum(0.05, 0.1 + 0.02 * np.sin(t / 10.0)))  # monotonically increasing-ish
    heading = (120 + 40 * np.sin(t / 25.0)) % 360
    pitch = -7 + 1.5 * np.sin(t / 15.0)
    return depth, heading, pitch


# =========================
# Training loop
# =========================
def train(cfg: TrainConfig, depth, heading, pitch, out_dir: str, ckpt_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    env = DriftEnv(depth=depth, heading=heading, pitch=pitch, cfg=cfg)
    n_states = 3
    n_actions = 1

    agent = DDPGAgent(n_states=n_states, n_actions=n_actions, cfg=cfg)

    epsilon = cfg.epsilon
    returns = []

    for ep in range(cfg.episodes):
        s = env.reset()
        ep_sum = 0.0

        traj = {
            "x": [], "y": [], "alpha": [], "reward": [], "dis": [],
            "dep": [], "heading": [], "pitch": []
        }

        for step in range(env.len):
            a = agent.choose_action(s, epsilon=epsilon)
            a = float(np.clip(a.squeeze(), env.a_low, env.a_high))

            s_, r, done = env.step(step, a)

            agent.store_transition(s, np.array([a], dtype=np.float32), r, s_)
            agent.learn()

            ep_sum += r
            s = s_

            traj["x"].append(env.x)
            traj["y"].append(env.y)
            traj["alpha"].append(a)
            traj["reward"].append(r)
            traj["dis"].append(env.dis)
            traj["dep"].append(float(depth[step]))
            traj["heading"].append(float(heading[step]))
            traj["pitch"].append(float(pitch[step]))

            if done:
                break

        if epsilon > cfg.epsilon_min:
            epsilon = max(cfg.epsilon_min, epsilon * cfg.epsilon_decay)

        returns.append(ep_sum)

        # Save trajectory per episode (CSV for portability)
        ep_path = os.path.join(out_dir, f"episode_{ep:04d}.csv")
        pd.DataFrame(traj).to_csv(ep_path, index=False)

    # Final save
    pd.DataFrame({"episode": np.arange(len(returns)), "reward": returns}).to_csv(
        os.path.join(out_dir, "rewards.csv"), index=False
    )
    agent.save(ckpt_dir)
    print(f"[OK] Saved checkpoints to: {ckpt_dir}")
    return returns

def infer(cfg: TrainConfig, depth, heading, pitch, out_dir: str, ckpt_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    env = DriftEnv(depth=depth, heading=heading, pitch=pitch, cfg=cfg)
    n_states, n_actions = 3, 1

    agent = DDPGAgent(n_states=n_states, n_actions=n_actions, cfg=cfg)
    agent.load(ckpt_dir=ckpt_dir, n_states=n_states)

    s = env.reset()
    traj = {"x": [], "y": [], "alpha": [], "reward": [], "dis": [], "dep": [], "heading": [], "pitch": []}

    ep_sum = 0.0
    for step in range(env.len):
        # inference: epsilon=0 -> pure policy
        a = agent.choose_action(s, epsilon=0.0)
        a = float(np.clip(a.squeeze(), env.a_low, env.a_high))

        s_, r, done = env.step(step, a)
        ep_sum += r
        s = s_

        traj["x"].append(env.x)
        traj["y"].append(env.y)
        traj["alpha"].append(a)
        traj["reward"].append(r)
        traj["dis"].append(env.dis)
        traj["dep"].append(float(depth[step]))
        traj["heading"].append(float(heading[step]))
        traj["pitch"].append(float(pitch[step]))

        if done:
            break

    out_path = os.path.join(out_dir, "inference_trajectory.csv")
    pd.DataFrame(traj).to_csv(out_path, index=False)
    print(f"[OK] Inference saved: {out_path}")
    print(f"[OK] Inference total reward: {ep_sum:.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default="train", choices=["train", "infer"],
                   help="Run mode: train or infer")
    p.add_argument("--data", type=str, default=None, help="Path to G_sample.xlsx or .csv")
    p.add_argument("--demo", action="store_true", help="Run with synthetic demo series")
    p.add_argument("--out_dir", type=str, default="outputs", help="Output directory")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints", help="Checkpoint directory")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()



def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = TrainConfig()
    if args.episodes is not None:
        cfg.episodes = int(args.episodes)

    # Load data
    if args.demo:
        depth, heading, pitch = make_demo_series()
    else:
        if not args.data:
            raise ValueError("Provide --data path or use --demo.")
        depth, heading, pitch = load_series(args.data)

    if args.mode == "train":
        train(cfg, depth, heading, pitch, out_dir=args.out_dir, ckpt_dir=args.ckpt_dir)
    else:
        infer(cfg, depth, heading, pitch, out_dir=args.out_dir, ckpt_dir=args.ckpt_dir)



if __name__ == "__main__":
    main()
