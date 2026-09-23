# gmml

**Reinforcement learning for GameMaker games.** An ML-Agents-style framework: your game stays
in GameMaker, the learning happens in Python, and the two talk over a local socket.

You describe an agent in GML (observations, actions, and rewards), point a YAML config at it, and run one command to train. When the policy is
good, export it back into the engine so a shipped game can run it without involving Python.

```
GameMaker (TCP server)                  Python (client)
    obj_gmml_academy    <-- JSON/TCP -->    gmml train
    obj_gmml_agent                          Gymnasium + Stable-Baselines3
```

The engine owns everything domain-specific. There is no per-game Python code to write.

---

## Requirements

- **GameMaker** (project built with IDE 2024.11, GML runtime)
- **Python 3.10+**

## Install

```bash
pip install -e .
```

Installs the `gmml` command and dependencies (gymnasium, stable-baselines3, numpy,
pyyaml, tensorboard).

## Quick start: the drone demo

The repo comes with a working example, a Box2D physics drone that learns to fly a waypoint course
without crashing into walls.

1. Open `gmproject/gmproject.yyp` in GameMaker.
2. Set **`rm_drone_train`** as the first room and press Run. You should see a grid of ~144 drone
   arenas sitting idle, waiting for a connection.
3. In a terminal, start training:

```bash
gmml train --config configs/drone.yaml
```

The drones start flying immediately. Training runs for 2M steps and writes everything to
`runs/drone/`. Press Ctrl+C at any point to stop training (the current policy is saved before exit).

Watch the learning curves with tensorboard:

```bash
tensorboard --logdir runs/drone/tensorboard
```

Then see the trained policy drive the game:

```bash
gmml infer --config configs/drone.yaml
```

In-game keys: `F6` debug overlay, `P` player override (fly the first drone manually with A/S/D/W),
`F12` quit.

---

## Commands

| Command | What it does |
| --- | --- |
| `gmml train --config C` | Train a behavior. `--resume` continues from the latest checkpoint, `--resume-from CKPT` from a specific one. |
| `gmml infer --config C` | Run a trained policy against a live GameMaker window. `--model PATH`, `--episodes N`. |
| `gmml export --config C` | Export the policy for in-engine inference. `--format gmpolicy` (default). |

Any config field can be overridden inline without having to edit the config file:

```bash
gmml train --config configs/drone.yaml --set hyperparameters.learning_rate=1e-4 --set total_timesteps=500000
```

## Config

A run is fully described by one YAML file. Anything not listed falls back to a default.

```yaml
behavior: drone              # must match behavior_name in GML
algo: PPO                    # PPO | A2C
total_timesteps: 2000000
vectorized: true             # many agents in one game window, one socket

hyperparameters:             # passed through to Stable-Baselines3
  policy: MlpPolicy
  net_arch: [64, 64]
  learning_rate: 0.0003
  n_steps: 256
  batch_size: 2304

self_play:
  enabled: false             # snapshot the learner, play it against its past selves

output:
  run_dir: runs/drone
  checkpoint_every: 100000

connection:
  host: localhost
  port: 5555
```

`vectorized: true` and `self_play.enabled: true` are mutually exclusive: vectorized training
resets each agent independently, self-play needs a paired global reset.

---

## Writing your own agent

Two objects do the work, both already in the project:

- **`obj_gmml_academy`** is the singleton transport. Place one in your training room and call
  `connect(5555)`. It finds agents automatically.
- **`obj_gmml_agent`** is the base object every behavior inherits from.

Create a child of `obj_gmml_agent` and fill in four things in its Create event:

```gml
event_inherited();
behavior_name = "drone";                  // match the behavior_name in your config
gmml_set_spec(17, gmml_continuous(2));    // 17 observations, 2 continuous actions
                                          // or gmml_discrete([3, 2]) for branched choices

collect_observations = function() {
    add_observation(clamp(phy_speed_x / maxSpeed, -1, 1));   // try to keep values roughly in [-1, 1]
    // ... 17 of these in the drone example
};

on_action = function(_action) {
    thrustL = _action[0];
    add_reward(progress_made);
    if (crashed) { add_reward(-5); end_episode(); }
};

agent_reset = function() {
    // re-init THIS agent for a new episode (vectorized mode)
};
```

Optional hooks: `agent_pause()` / `agent_resume()` let you handle Python starting/ending a gradient update. Nice for physics environments like the drone example to pause/resume physics so agents don't drift into junk states between rollouts which would soil the training data. The academy's
`on_reset` handles whole-scene rebuilds when you aren't using per-agent resets.

Then write a config with a matching `behavior:` and train.

## Shipping a trained policy

`gmml infer` needs Python running alongside the game (duh), which is fine for development but not really viable for
a release build. Export instead:

```bash
gmml export --config configs/drone.yaml
```

This writes `runs/drone/final_policy.gmpolicy`, a small binary holding the network weights and
action spec. Add it to the project as an Included File and load it with the pure-GML runtime in
`scr_gmml_inference` to run the model entirely in-engine:

```gml
policy = gmpolicy_load("final_policy.gmpolicy");
// each step
gmml_policy_step(policy);
```

`rm_drone_infer` is a working example of this, driven by `obj_gmml_runtime`.

## Project layout

```
gmml/           Python package: CLI, env wrappers, protocol, training, export
configs/        behavior YAML files
gmproject/      GameMaker project: the gmml GML runtime + the drone demo
  scripts/scr_gmml_*          protocol, agent API, in-engine inference
  objects/obj_gmml_*          academy (transport), agent (base object), runtime (inference)
  objects/obj_agent_drone     the demo behavior, a good reference to have a look at
tests/          pytest suite, runs against a mock engine (no GameMaker needed)
runs/           training output: checkpoints, tensorboard logs, final policy
```

Run the tests with `pip install -e ".[dev]"` followed by `pytest tests`. The suite drives a mock
engine, so GameMaker does not need to be running.

## Status

Early but working. Known gaps:

- PPO and A2C only.
- `.gmpolicy` inference is deterministic only (continuous returns the distribution mean,
  discrete returns per-branch argmax), matching `gmml infer`.
- One GameMaker window per training run. Vectorization comes from many agents inside that one
  window, not many windows.
