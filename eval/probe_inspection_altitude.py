"""Physical energy-control falsification on fixed shared timed programs.

Four aircraft climb to96m during a turn, then descend to their home levels.
No actor, target or policy; this isolates phase errors caused by altitude steps.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from envs.inspection.control import ControlGains,FixedwingAutopilot
from envs.inspection.flight import FlightWorld
from envs.inspection.paths import Pose,RouteProgram,TimedPath,capsule_from_bottom
from envs.inspection.planning import recovery


def probe(gains,wind,descent_at=65.,duration=160.):
    flight=FlightWorld(0,jitter=False,wind_xy=wind)
    programs=[]
    for home in flight.centers:
        search=TimedPath(capsule_from_bottom(Pose(-250.,home-80.,0.),home),wind,18.)
        returning,tail=recovery(search.path[-1].end,home,wind)
        programs.append(RouteProgram(TimedPath.concatenate((search,returning)),tail))
    flight.controllers=[FixedwingAutopilot(gains) for _ in range(4)]
    audit={"physics_samples":0,"max_xy_error_m":0.,"max_settled_z_error_m":0.,
           "xy_6m_breach_steps":0,"settled_z_2m_breach_steps":0,"peak_xy_witness":None}
    audit["vertical_transition_envelope_breach_steps"]=0
    original=flight.world.after_physics
    settle=20. if gains.max_vertical_speed is None else 24/gains.max_vertical_speed+20
    def after(stamp):
        original(stamp)
        positions=[flight.world.getBasePositionAndOrientation(d.Id)[0] for d in flight.world.drones]
        audit["physics_samples"]+=1
        for i,(pos,p) in enumerate(zip(positions,programs)):
            error=float(np.linalg.norm(np.asarray(pos[:2])-p.at(stamp)[:2]))
            if error>audit["max_xy_error_m"]:
                audit["max_xy_error_m"]=error
                audit["peak_xy_witness"]={"t":stamp,"drone":i,"actual_position":pos,"nominal":p.at(stamp).tolist()}
        audit["xy_6m_breach_steps"]+=int(any(np.linalg.norm(np.asarray(pos[:2])-p.at(stamp)[:2])>6 for pos,p in zip(positions,programs)))
        phase_start,target=(descent_at,flight.altitudes) if stamp>=descent_at else ((30.,[96.]*4) if stamp>=30 else (0.,flight.altitudes))
        old=([96.]*4 if stamp>=descent_at else flight.altitudes)
        vertical_bad=any(not min(previous,goal)-2 <= pos[2] <= max(previous,goal)+2
                         for pos,previous,goal in zip(positions,old,target))
        audit["vertical_transition_envelope_breach_steps"]+=int(vertical_bad)
        if stamp-phase_start >= (20. if phase_start == 0 else settle):
            maximum=max(abs(pos[2]-alt) for pos,alt in zip(positions,target))
            audit["max_settled_z_error_m"]=max(audit["max_settled_z_error_m"],maximum)
            audit["settled_z_2m_breach_steps"]+=int(maximum>2)
    flight.world.after_physics=after
    trace,error=[],None
    try:
        while flight.t<duration-1e-8:
            altitudes=[96.]*4 if 30<=flight.t<descent_at else flight.altitudes
            targets=[p.target(flight.t,flight.world.state(i)[3],float(altitudes[i])) for i,p in enumerate(programs)]
            flight.step(targets,exit_flags=[flight.t>=p.prefix.duration for p in programs])
            if flight.world.physics_steps%24==0:
                trace.append({"t":flight.t,"positions":[s[3].tolist() for s in flight.world.all_states]})
    except Exception:
        error=traceback.format_exc()
    finally:
        safety=flight.summary()
        try:flight.close()
        except Exception:error=(error or "")+traceback.format_exc()
    return {"role":"development_energy_control_falsification_not_full_qualification","control_gains":asdict(gains),
            "wind_xy_m_s":wind,"altitude_schedule":{"climb_at":30.,"descend_at":descent_at,"high_m":96.,"otherwise":"home72/80/88/96"},
            "settle_seconds_after_altitude_step":settle,"duration_seconds":duration,
            "passed":not error and not safety["violations"] and audit["xy_6m_breach_steps"]==0
            and audit["settled_z_2m_breach_steps"]==0 and audit["vertical_transition_envelope_breach_steps"]==0
            and all(t>=20 for t in safety["exit_dwell_seconds"]),
            "tracking_240hz":audit,"safety":safety,"operational_error":error,"trace":trace}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--altitude-throttle-p",type=float,default=0.)
    parser.add_argument("--speed-p",type=float,default=.08)
    parser.add_argument("--vertical-speed-limit",type=float)
    parser.add_argument("--descent-at",type=float,default=65.)
    parser.add_argument("--seconds",type=float,default=160.)
    parser.add_argument("--wind",type=float,nargs=2,default=[0.,0.])
    args=parser.parse_args()
    if args.out.exists():parser.error("choose a new output")
    sources={str(p.relative_to(ROOT)):p.read_text() for p in [*sorted((ROOT/"envs/inspection").glob("*.py")),Path(__file__)]}
    gains=replace(ControlGains(),pitch_p=4.5,elevator_trim=-.1,speed_p=args.speed_p,
                  altitude_throttle_p=args.altitude_throttle_p,max_vertical_speed=args.vertical_speed_limit)
    try:result=probe(gains,args.wind,args.descent_at,args.seconds)
    except Exception:result={"passed":False,"operational_error":traceback.format_exc()}
    result["source_snapshots"]=sources
    result["source_hashes"]={p:hashlib.sha256(s.encode()).hexdigest() for p,s in sources.items()}
    result["sources_unchanged"]=all((ROOT/p).read_text()==s for p,s in sources.items())
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("trace","source_snapshots","source_hashes")},indent=2))
    sys.exit(0 if result["passed"] and result["sources_unchanged"] else 1)
