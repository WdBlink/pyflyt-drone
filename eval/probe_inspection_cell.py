"""Bounded physical falsifier for fixed arcs with finite curvature ramps.

Development only. The fixed40m runout is a test maneuver, not a target forecast.
Uses the real Mission camera/transport/report path, no model or wallclock claim.
"""

from dataclasses import asdict
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.inspection.mission import InspectionMission
from envs.inspection.paths import Pose, RouteProgram, TimedPath, course_kinematics, fleet_clearance
from envs.inspection.planning import history_prefix, recovery
from envs.inspection.scenario import Entity


def maneuver(sample, wind):
    """Two fixed arcs, a40m tangent runout, and explicit2s curvature ramps."""
    xy = np.asarray(sample[:2]).copy()
    heading = float(sample[2])
    rows, stamp = [], 0.
    for seconds, first_k, last_k in ((2., 0., 1/80.), (26., 1/80., 1/80.),
                                     (2., 1/80., 0.), (40/18., 0., 0.),
                                     (2., 0., 1/80.), (5., 1/80., 1/80.),
                                     (2., 1/80., 0.)):
        elapsed = 0.
        while elapsed < seconds-1e-9:
            step = min(.025, seconds-elapsed)
            u = elapsed/seconds
            k = first_k+(last_k-first_k)*(3*u*u-2*u*u*u)
            speed, bank = course_kinematics(heading, k, np.asarray(wind), 18.)
            rows.append(np.r_[stamp, xy, heading, speed, bank])
            mid_u = (elapsed+step/2)/seconds
            mid_k = first_k+(last_k-first_k)*(3*mid_u**2-2*mid_u**3)
            mid_heading = heading+.5*step*speed*mid_k
            mid_speed, _ = course_kinematics(mid_heading, mid_k, np.asarray(wind), 18.)
            xy += step*mid_speed*np.array([math.cos(mid_heading), math.sin(mid_heading)])
            heading += step*mid_speed*mid_k
            elapsed += step
            stamp += step
    speed, bank = course_kinematics(heading, 0., np.asarray(wind), 18.)
    rows.append(np.r_[stamp, xy, heading, speed, bank])
    return TimedPath.from_samples(rows, wind, 18.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('choose a new output')
    args.out.mkdir(parents=True)
    frozen = {str(p.relative_to(ROOT)): p.read_text() for p in sorted((ROOT/'envs/inspection').glob('*.py'))}
    started = time.monotonic()
    mission = InspectionMission(11, reference=True, events=False, start_planner=False, wind_xy=(-2., 0.))
    selected = 1
    body = next(iter(mission.entity_bodies.values()))
    for other in mission.entity_bodies.values():
        mission.flight.world.resetBasePositionAndOrientation(other, [0, 0, -10], [0, 0, 0, 1])
    entity = Entity('cell-fixture', 'dynamic', 'inspection_vehicle', (-250., -100.), 0., 1.5, 'dynamic-inspection')
    mission.entities, mission.entity_bodies = [entity], {entity.key: body}
    candidate, pending, geometry = None, None, None
    adopted, reported = False, False
    distances, witnesses = [], []
    last_public = -1.
    history = []
    try:
        while mission.t < 58.-1e-8:
            if mission.flight.world.physics_steps % 120 == 0:
                state = mission.public_state(since=last_public)
                last_public = mission.t
                history.extend(state['observations'])
                version = state['aircraft'][selected]['job_version']
                if candidate is None:
                    seed = next((o for o in history if o['drone_id'] == selected and o['object_type'] == 'dynamic'), None)
                    if seed:
                        candidate = seed['candidate_id']
                        ack = mission.tool('inspect_candidate', dict(drone_id=selected, candidate_id=candidate,
                            expected_version=version, rule_id='dynamic-inspection', task_card_version=1))
                        if ack['status'] != 'accepted':
                            raise RuntimeError(ack)
                        commit = mission.t+.5
                        old = mission.executor.programs[selected]
                        work = maneuver(old.at(commit), mission.executor.wind)
                        returning, tail = recovery(Pose(*work.at(work.duration)[:3]), -100., mission.executor.wind)
                        program = RouteProgram(TimedPath.concatenate((history_prefix(old, commit), work, returning)), tail)
                        programs = list(mission.executor.programs)
                        programs[selected] = program
                        geometry = fleet_clearance(programs, sample_dt=.1)
                        if geometry['continuous_lower_bound_horizontal_m'] < 27:
                            raise RuntimeError('nominal traffic conflict before physical test')
                        pending = (commit, tuple(programs), work.duration)
                confirmed = any(j['kind'] == 'inspect' and j['status'] == 'succeeded' for j in state['jobs'])
                qualified = [o for o in history if o['candidate_id'] == candidate and o['quality'] >= .8 and o['sensor_mode'] == 'inspect']
                if confirmed and qualified and not reported:
                    point = qualified[-1]
                    request = {k: point[k] for k in ('object_type', 'candidate_id', 'measured_at', 'estimate_xy')}
                    request.update(report_id='cell-report', claimed_class=point['estimated_class'], evidence_ids=[point['evidence_id']])
                    if mission.tool('submit_or_revise_report', request)['status'] == 'accepted':
                        ack = mission.tool('observe_track', dict(drone_id=selected, candidate_id=candidate, duration=180.,
                            report_id='cell-report', expected_version=version, rule_id='dynamic-inspection', task_card_version=1))
                        reported = ack['status'] == 'accepted'
            if pending and not adopted and mission.t >= pending[0]-1e-8:
                capture, metrics = mission.executor._capture(selected, mission.t, mission.flight.world.all_states)
                if not capture:
                    raise RuntimeError({'capture': metrics})
                mission.executor.programs = pending[1]
                mission.executor.audit.append(dict(type='development_fixture_program_adopted', at=mission.t,
                    work_seconds=pending[2], no_asynchronous_or_qualification_claim=True))
                adopted = True
            mission.step()
            if mission.flight.world.physics_steps % 24 == 0:
                own = mission.flight.world.state(selected)[3]
                target = entity.position(mission.t)
                distance = float(np.linalg.norm(own[:2]-target))
                distances.append(distance)
                witnesses.append(dict(t=mission.t, own=own.tolist(), truth=target,
                    nominal=mission.executor.programs[selected].at(mission.t).tolist(), horizontal_distance_m=distance))
        business = mission.business_audit()
        result = dict(role='single_development_ramp_capture_falsifier_not_G1_G3_or_full_180s_task',
            simulated_seconds=mission.t, wall_seconds=time.monotonic()-started, source_files=frozen,
            sources_unchanged=all((ROOT/p).read_text() == s for p,s in frozen.items()),
            source_sha256={p:hashlib.sha256(s.encode()).hexdigest() for p,s in frozen.items()},
            fixture=dict(root=11, selected_aircraft=selected, wind_xy=[-2., 0.], fixed_runout_m=40.),
            adopted=adopted, report_and_observe_accepted=reported, geometry=geometry,
            max_sampled_optical_horizontal_distance_m=max(distances), witnesses=witnesses,
            limitations=['fixed test maneuver; no causal recentering policy', 'no 180s success or new tube claim',
                'controlled future adoption, no asynchronous1x timing', '58s may not finish the complete reserved return'], **business)
        (args.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k:result[k] for k in ('adopted','report_and_observe_accepted','simulated_seconds',
            'max_sampled_optical_horizontal_distance_m','execution_tracking_240hz','safety')}, indent=2))
    finally:
        mission.close()


if __name__ == '__main__':
    main()
