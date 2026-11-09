"""Controlled finite-arc task then permanent planner stop and physical return.

One development fixture; not an asynchronous1x or full180s qualification.
"""

import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import envs.inspection.mission as mission_module
from envs.inspection.scenario import Entity
from envs.inspection.control import FixedwingAutopilot,inspection_gains
from eval.qualify_inspection_executor import ControlledPlannerExecutor


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--pitch-trim',type=float)
    parser.add_argument('--root',type=int,default=11)
    parser.add_argument('--selected',type=int,choices=range(4),default=1)
    parser.add_argument('--wind',type=float,nargs=2,default=[-2.,0.])
    args=parser.parse_args()
    if args.out.exists():parser.error('choose a new output')
    args.out.mkdir(parents=True)
    paths=[*sorted((ROOT/'envs/inspection').glob('*.py')),Path(__file__),ROOT/'eval/qualify_inspection_executor.py']
    sources={str(p.relative_to(ROOT)):p.read_text() for p in paths}
    original=mission_module.FlightExecutor
    mission_module.FlightExecutor=ControlledPlannerExecutor
    try:mission=mission_module.InspectionMission(args.root,reference=True,events=False,wind_xy=tuple(args.wind))
    finally:mission_module.FlightExecutor=original
    started=time.monotonic()
    selected=args.selected
    candidate=None
    reported=stopped=False
    last_public=-1.
    history=[]
    try:
        gains=inspection_gains() if args.pitch_trim is None else replace(inspection_gains(),pitch_trim=args.pitch_trim)
        mission.flight.controllers=[FixedwingAutopilot(gains) for _ in range(4)]
        body=next(iter(mission.entity_bodies.values()))
        for other in mission.entity_bodies.values():
            mission.flight.world.resetBasePositionAndOrientation(other,[0,0,-10],[0,0,0,1])
        entity=Entity('cell-fixture','dynamic','inspection_vehicle',(-250.,float(mission.flight.centers[selected])),0.,1.5,'dynamic-inspection')
        mission.entities,mission.entity_bodies=[entity],{entity.key:body}
        for i in (i for i in range(4) if i!=selected):
            ack=mission.tool('enter_safe_exit',dict(drone_id=i,expected_version=0))
            if ack['status']!='accepted':raise RuntimeError(ack)
        while mission.t<150.-1e-8:
            if mission.flight.world.physics_steps%120==0:
                state=mission.public_state(since=last_public)
                last_public=mission.t
                history.extend(state['observations'])
                version=state['aircraft'][selected]['job_version']
                if candidate is None:
                    seed=next((o for o in history if o['drone_id']==selected and o['object_type']=='dynamic'),None)
                    if seed:
                        candidate=seed['candidate_id']
                        ack=mission.tool('inspect_candidate',dict(drone_id=selected,candidate_id=candidate,
                            expected_version=version,rule_id='dynamic-inspection',task_card_version=1))
                        if ack['status']!='accepted':raise RuntimeError(ack)
                confirmed=any(j['kind']=='inspect' and j['status']=='succeeded' for j in state['jobs'])
                qualified=[o for o in history if o['candidate_id']==candidate and o['quality']>=.8 and o['sensor_mode']=='inspect']
                if confirmed and qualified and not reported:
                    point=qualified[-1]
                    request={k:point[k] for k in ('object_type','candidate_id','measured_at','estimate_xy')}
                    request.update(report_id='cell-report',claimed_class=point['estimated_class'],evidence_ids=[point['evidence_id']])
                    if mission.tool('submit_or_revise_report',request)['status']=='accepted':
                        ack=mission.tool('observe_track',dict(drone_id=selected,candidate_id=candidate,duration=180.,
                            report_id='cell-report',expected_version=version,rule_id='dynamic-inspection',task_card_version=1))
                        reported=ack['status']=='accepted'
            if mission.t>=18. and not stopped:
                executor=mission.executor
                executor.process=None
                executor.pending=executor.ready=executor.controlled_response=None
                executor.pending_wall_deadline=None
                executor.audit.append(dict(type='controlled_permanent_planner_stop',at=mission.t,
                    navigation=[s.tolist() for s in mission.flight.world.all_states],
                    world_velocities=[mission.flight.world.getBaseVelocity(d.Id)[0] for d in mission.flight.world.drones],
                    controller_integrals=[(c.altitude_integral,c.speed_integral) for c in mission.flight.controllers],
                    limitation='controlled transport; not asynchronous wallclock evidence'))
                stopped=True
            mission.step()
        result=dict(role='one_controlled_capture_and_complete_recovery_falsifier_not_full_qualification',
            fixture=dict(root=args.root,selected_aircraft=selected,wind_xy=args.wind),actual_control_gains=asdict(gains),
            simulated_seconds=mission.t,wall_seconds=time.monotonic()-started,source_files=sources,
            sources_unchanged=all((ROOT/p).read_text()==s for p,s in sources.items()),
            source_sha256={p:hashlib.sha256(s.encode()).hexdigest() for p,s in sources.items()},
            report_and_observe_accepted=reported,planner_stopped=stopped,**mission.business_audit())
        result['passed_bounded_fixture']=(reported and stopped and not result['safety']['violations']
            and not result['execution_infrastructure_errors']
            and result['execution_tracking_240hz']['horizontal_tube_breach_steps']==0
            and result['execution_tracking_240hz']['altitude_envelope_breach_steps']==0
            and result['execution_tracking_240hz']['seam_capture_failures']==0
            and all(t>=20 for t in result['safety']['exit_dwell_seconds']))
        (args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:result[k] for k in ('passed_bounded_fixture','report_and_observe_accepted',
            'execution_tracking_240hz','safety')},indent=2))
        return 0 if result['passed_bounded_fixture'] else 1
    finally:mission.close()


if __name__=='__main__':sys.exit(main())
