# Bounded primary-source literature matrix

Search run and update cutoff: 2026-08-05 (Asia/Shanghai), inclusive. This is a
bounded screen derived from the R1.3 literature-search artifact and normalized
in `ctx/scratch/02-systematic-literature-screen.md`. It is not a systematic
review, priority claim, matched reproduction, or SOTA conclusion. `unresolved`
means the retrieved primary page did not establish the detail.

| Work | Task relation | Deployment truth access evidence | Runnable status | Protocol role |
|---|---|---|---|---|
| Wang et al. (2024), [vision-based DRL with privileged information](https://arxiv.org/abs/2412.06313) | UAV visual navigation | seed review records actor target-distance input | unresolved | context |
| Chai et al. (2024), [fixed-wing monocular obstacle avoidance](https://arxiv.org/abs/2411.18009) | nearest fixed-wing PPO-family perception work | seed review records relative target distance/angle | code reported by seed review; URL unresolved | matched-reproduction candidate |
| Liou & Cheng (2026), [fixed-wing tracking/terminal guidance](https://arxiv.org/abs/2607.12801) | target tracking with YOLO/UKF/CBF, not RL actor | controller access unresolved | unresolved | context/control candidate |
| Zhang et al. (2026), [lightweight safe RL](https://arxiv.org/abs/2607.01794) | constraint/curriculum context | unresolved | unresolved | context |
| Tai et al. (2023), [PyFlyt](https://arxiv.org/abs/2304.01305) | simulator platform | n/a | [official code](https://github.com/jjshoots/PyFlyt) | platform |
| Pothuri et al. (2025), [vision-based active tracking](https://arxiv.org/abs/2506.18264) | air-to-air visual tracking/redetection | no world coordinate stated in retrieved evidence | unresolved | context/comparison candidate |
| Sun et al. (2025), [DAT/GC-VAT](https://github.com/SHWplus/DAT_Benchmark) | open-world drone tracking benchmark | unresolved | official project repository retrieved | matched-reproduction candidate |
| Gong et al. (2025), [VVC-Gym/Fly-Craft](https://github.com/GongXudong/fly-craft) | fixed-wing state/goal-vector control | desired velocity goal is an observation/API goal | official project repository retrieved | platform/comparison candidate |
| Wiedemann et al. (2023), [analytic policy gradient](https://github.com/lis-epfl/apg_trajectory_tracking) | fixed-wing trajectory control | reference-state access; target contract unresolved | official repository retrieved | comparison candidate |
| Pham et al. (2018), [autonomous UAV navigation](https://arxiv.org/abs/1801.05086) | navigation context | unresolved | unresolved | context |
| Madaan et al. (2020), [AirSim Drone Racing Lab](https://proceedings.mlr.press/v119/madaan20a.html) | multirotor vision racing | unresolved | [official code](https://github.com/microsoft/AirSim-Drone-Racing-Lab) | context |
| He et al. (2021), [explainable UAV DRL](https://github.com/heleidsn/UAV_Navigation_DRL_AirSim) | AirSim navigation/path planning | unresolved | project repository retrieved | context |

The frozen conservative boundary is: no retrieved evidence establishes absence
of prior combinations, novelty, superiority, or reproducibility. Until a
predeclared matched-reproduction matrix is executed and independently reviewed,
the maximum performance claim remains “best among the frozen evaluated internal
methods.”
