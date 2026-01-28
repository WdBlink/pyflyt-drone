#!/usr/bin/env python3
"""
Quick smoke test for ArdupilotGazeboObjLockEnv.

Expected usage:
  - Start gz sim + ArduPilot SITL + MAVROS (and optionally ros_gz_bridge for images).
  - Then run:
      python scripts/test_ardupilot_ros_objlock_env.py

This script prints decoded attitude components and aux_state (servo outputs) periodically,
so you can verify your subscriptions are alive and updating.
"""

from __future__ import annotations

import os
import sys
import argparse
import time

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from envs.ardupilot_ros_objlock_env import ArdupilotGazeboObjLockEnv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hz", type=int, default=5, help="Print rate (Hz)")
    ap.add_argument("--duration", type=float, default=30.0, help="How long to run (seconds)")
    ap.add_argument("--enable-rc", action="store_true", help="Enable RC override publishing")

    ap.add_argument("--imu-topic", type=str, default="/mavros/imu/data")
    ap.add_argument("--odom-topic", type=str, default="/mavros/local_position/odom")
    ap.add_argument("--vel-topic", type=str, default="/mavros/local_position/velocity_local")
    ap.add_argument("--servo-topic", type=str, default="/mavros/servo_output/raw")
    ap.add_argument("--rc-out-topic", type=str, default="/mavros/rc/out")
    ap.add_argument("--rc-override-topic", type=str, default="/mavros/rc/override")

    ap.add_argument("--rgb-topic", type=str, default="", help="Optional ROS Image topic for RGB")
    ap.add_argument("--depth-topic", type=str, default="", help="Optional ROS Image topic for depth")

    args = ap.parse_args()

    env = ArdupilotGazeboObjLockEnv(
        angle_representation="euler",
        agent_hz=30,
        max_duration_seconds=120.0,
        enable_rc_override=bool(args.enable_rc),
        mavros_imu_topic=args.imu_topic,
        mavros_odom_topic=args.odom_topic,
        mavros_vel_topic=args.vel_topic,
        mavros_servo_output_topic=args.servo_topic,
        mavros_rc_out_topic=args.rc_out_topic,
        mavros_rc_override_topic=args.rc_override_topic,
        rgb_image_topic=args.rgb_topic or None,
        depth_image_topic=args.depth_topic or None,
        # Defaults match your mini_talon_vtail model.sdf comments:
        servo_ch_aileron=1,
        servo_ch_vtail_left=2,
        servo_ch_throttle=3,
        servo_ch_vtail_right=4,
        split_aileron_left_right=True,
        vtail_mix_to_elev_rudder=False,
    )

    obs, info = env.reset()
    print("reset info:", info)
    print("attitude dim:", obs["attitude"].shape, "duck_vision dim:", obs["duck_vision"].shape)

    print_dt = 1.0 / max(1, int(args.hz))
    t_end = time.monotonic() + float(args.duration)
    t_next = time.monotonic()

    # Passive stepping is still useful: it drives spin_once() and (optionally) control publishing.
    a = np.zeros((4,), dtype=np.float32)
    i = 0
    while time.monotonic() < t_end:
        obs, reward, term, trunc, info = env.step(a)
        if time.monotonic() >= t_next:
            snap = env.get_debug_snapshot()
            att = obs["attitude"].astype(np.float32).reshape(-1)
            # Euler attitude layout: [ang_vel(3), euler(3), lin_vel(3), lin_pos(3), prev_action(4), aux_state(6)]
            ang_vel = att[0:3]
            euler = att[3:6]
            lin_vel = att[6:9]
            lin_pos = att[9:12]
            aux = att[-6:]

            servo_pwm = snap.get("servo_pwm")
            servo_pwm_used = snap.get("servo_pwm_used_for_aux")
            servo2 = None
            servo1 = None
            servo4 = None
            if isinstance(servo_pwm, np.ndarray) and servo_pwm.size >= 4:
                servo2 = int(servo_pwm[1])
                servo1 = int(servo_pwm[0])
                servo4 = int(servo_pwm[3])

            servo2_used = None
            servo4_used = None
            if isinstance(servo_pwm_used, np.ndarray) and servo_pwm_used.size >= 4:
                servo2_used = int(servo_pwm_used[1])
                servo4_used = int(servo_pwm_used[3])

            print(
                f"[{i:05d}] connected={info.get('mavros_connected')} imu={info.get('have_imu')} pos={info.get('have_pos')} vel={info.get('have_vel')} servo={info.get('have_servo')} "
                f"ang_vel={ang_vel.round(3).tolist()} euler={euler.round(3).tolist()} "
                f"lin_vel={lin_vel.round(3).tolist()} lin_pos={lin_pos.round(3).tolist()} "
                f"servo2_raw={servo2} servo1_raw={servo1} servo4_raw={servo4} used2={servo2_used} used4={servo4_used} aux_state={aux.round(3).tolist()}"
            )
            t_next += print_dt
        i += 1

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
