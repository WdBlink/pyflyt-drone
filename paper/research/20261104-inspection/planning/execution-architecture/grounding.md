# Shared executor grounding

This is authorized prerequisite engineering for the existing RA-L inspection plan. No formal test/holdout data, model API or publication is used.

## Current ownership and flow

`eval/run_inspection_reference.py` launches a worker owning `InspectionMission`; the coordinator receives serialized public snapshots and sends typed tool envelopes. Unbounded producer queues keep the physical worker independent of coordinator waits. The worker steps one shared four-aircraft PyFlyt world at240 Hz physics/120 Hz control. Every physics integration is audited.

`envs/inspection/flight.py` owns each aircraft navigation target, conventional controllers, capsule routes and safe-exit circles. `control.py` maps heading/altitude/airspeed to six surfaces, bounds bank to30 degrees and uses crab correction from shared ideal local airdata. No mission target truth reaches control. Uniform0/2 m/s wind, flat terrain and the original five control families passed100/100 development slots with independent audit. Existing qualification does not cover new modes.

`mission.py` owns hidden entities/synthetic cameras, causal local trackers, sensor/report transport, public card, job execution/postconditions and actor RPC policy. `protocol.py` owns job versions and immutable evidence/report admission. `scoring.py` owns evidence selection and localization quality. The mission currently sets aircraft modes directly; this boundary can be deepened, but hidden truth must stay out of route/track inputs.

Public survey regions are four horizontal stripes with centers y=-300,-100,100,300 in an800x800 irregular ROI. Allowed flight square is[-550,550]^2. Standard capsule has half-length250 m and radius80 m; left/right arcs extend to x=+-330 m, y=center+-80 m. Exit circles center(0,stripe_center), radius80 m. Aircraft start at lower-left capsule points headingeast, altitudes72/80/88/96 m. Qualified envelope is AGL60..100, airspeed15..25, abs bank<=45 hard audit (controller target bank<=30), min3D separation15 m. Nominal airspeed20 m/s. Explicit distance budget18000 m and exit reserve1000 m.

The first600 s 1x B0 reference completed with C=.928,S=1,D=.00294,H=E=1,Y=.643655,SR=0. A5 s coordinator block advanced physics4.9917 s; maxlag.0339 s. Executed-journal replay matched13 artifact groups exactly. It is one dev reference, not a formal comparison. Its weak dynamic result revealed duplicate inspection before proof arrival and no observe_track invocation. Current observe jobs would narrow/point the camera but leave the survey route unchanged. Current assign_survey rejects a stripe different from the aircraft index, preventing general reassignment. These gaps block full G2/G3/G4.

## Required next behavior

Keep the actor API at mission tasks: assign_survey(any declared region), inspect_candidate(observed handle), observe_track(observed handle,duration), handoff_observation, cancel_job, enter_safe_exit. ACK is queued, completion uses physical/sensor postconditions. Past observations/own navigation/declared map are allowed. Hidden coordinates/counts/IDs/seed/future/events/grader correctness are not route inputs. Follow a moving candidate using only causal local measurements, tolerate temporary sensing loss without querying truth, and record gaps. Cross-region transit must actually fly bounded turns and remain inside the fence; traffic conflict may queue/reject with public reasons but must not silently pause physics or automatically solve mission allocation. Cancellation, stale versions, event retirement and deadline failures must remain auditable.

Use upstream PyFlyt. Reuse a known bounded-curvature/path primitive if useful, with primary-source/license verification before implementation. Do not add a large framework, train PPO, teleport aircraft, grant arbitrary hover/backward moves or weaken the scientific task to fixed stripes. Plan tests before implementation, including dense/four-aircraft cases and fresh qualifications for new modes. Human domain review remains pending.
