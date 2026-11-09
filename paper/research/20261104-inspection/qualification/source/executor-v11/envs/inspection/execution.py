"""Single job/flight-program owner. Planning never blocks the physical tick.

This is a DEVELOPMENT executor until the new mode qualification is complete.
The public actor has no reference to this object or its internal queues.
"""

from __future__ import annotations

from dataclasses import asdict, fields
import hashlib
import math
import multiprocessing as mp
import pickle
import queue
import time

import numpy as np

from envs.inspection.control import SurveyRoute, air_heading_for_course, inspection_gains, wrap_angle
from envs.inspection.guidance import LocalTrack
from envs.inspection.planning import (ObservedTrack, PlanProposal, PlanRequest, initial_program, plan)
from envs.inspection.protocol import JobBook


def program_hash(program):
    digest = hashlib.sha256()
    for path in (program.prefix,program.tail):
        digest.update(path.samples.tobytes())
        digest.update(path.wind.tobytes())
    return digest.hexdigest()


def planning_worker(requests,results):
    while True:
        request = requests.get()
        if request is None:
            return
        started = time.monotonic()
        try:
            proposal = plan(request)
        except (ValueError,ArithmeticError) as error:
            proposal = PlanProposal(request.serial,request.revision,request.commit_at,request.drone,request.tokens,
                                    None,None,None,None,str(error))
        payload = pickle.dumps((proposal,time.monotonic()-started),protocol=5)
        if len(payload) > 1100000:
            proposal = PlanProposal(request.serial,request.revision,request.commit_at,request.drone,request.tokens,
                                    None,None,None,None,"proposal exceeds bounded internal message size")
            payload = pickle.dumps((proposal,time.monotonic()-started),protocol=5)
        results.put(payload)


class FlightExecutor:
    """Three responsibilities behind one writer: jobs, local tracks, programs.

    Mission owns sensors/transport/reports. Samples accepted here carry no
    hidden identity or truth; per-aircraft local tracks are never cross-fused.
    """

    def __init__(self,homes,altitudes,wind,*,start_planner=True):
        if len(homes) != 4 or len(altitudes) != 4:
            raise ValueError("exactly four aircraft required")
        self.homes,self.home_altitudes,self.wind = tuple(homes),tuple(altitudes),tuple(wind)
        self.book = JobBook(4)
        self.programs = tuple(initial_program(float(y),self.wind) for y in homes)
        self.work_altitudes = list(altitudes)
        self.vertical_command_limit = inspection_gains().max_vertical_speed
        self.capture_lead_seconds = .5
        self.inspect_camera_trial_seconds = 2.
        self.altitude_transitions = [(float(a),float(a),0.) for a in altitudes]
        self.local = [dict() for _ in homes]
        self.seen_local = [set() for _ in homes]
        self.support = [dict() for _ in homes]
        self.revision,self.serial = 0,0
        self.pending,self.ready = None,None
        self.pending_wall_deadline = None
        self.expired_serials = set()
        self.retry_after = {}
        self.refreshed_at = {}
        self.views,self.service_samples = {},{}
        self.survey_monitors = {}
        self.audit = []
        self.infrastructure_errors = []
        self.envelope_id = "development-unqualified-18m_s-1p2_vertical-command-v3"
        self.tracking = {"physics_samples":0,"max_horizontal_error_m":0.,"max_settled_altitude_error_m":0.,
                         "horizontal_tube_breach_steps":0,"altitude_envelope_breach_steps":0}
        self.context = mp.get_context("spawn")
        self.requests,self.results = self.context.Queue(maxsize=1),self.context.Queue(maxsize=1)
        self.process = None
        if start_planner:
            self.process = self.context.Process(target=planning_worker,args=(self.requests,self.results),daemon=True)
            self.process.start()

    def active(self,i):
        job = self.book.jobs.get(self.book.active.get(i))
        return job if job and job.status in ("queued","running") else None

    def tokens(self):
        return tuple((self.book.versions[i],self.book.active.get(i)) for i in range(4))

    def submit(self,drone,kind,params,expected_version,now,*,seed=None,preempt=False):
        params = dict(params)
        if seed is not None:
            if seed.arrived_at > now or seed.measured_at > seed.arrived_at:
                raise ValueError("only delivered seed may cross to another aircraft")
            params["object_type"] = seed.object_type
        response = self.book.create(drone,kind,params,expected_version,now,preempt=preempt)
        if response["status"] == "accepted" and kind == "inspect":
            # Try two real camera views on the current committed route before
            # spending a long transit on a target already in the footprint.
            self.retry_after[response["job_id"]] = now+self.inspect_camera_trial_seconds
        if seed is not None and response["status"] == "accepted":
            candidate = params["candidate_id"]
            if candidate not in self.local[drone]:
                self.local[drone][candidate] = LocalTrack.first(seed.estimate_xy,seed.measured_at)
                self.support[drone][candidate] = (seed.evidence_id,seed.arrived_at)
        return response

    def cancel(self,job_id,expected_version,now):
        # Logical cancellation cannot free the already committed physical path.
        return self.book.cancel(job_id,expected_version,now)

    def mark_handoff(self,destination_job,source):
        destination_job.params["handoff_from_job"] = source.job_id
        destination_job.params["handoff_from_version"] = source.version

    def local_measurement(self,obs,now):
        if (not 0 <= obs.measured_at <= now or type(obs.drone_id) is not int or not 0 <= obs.drone_id < 4):
            raise ValueError("causal local sample required")
        i,candidate = obs.drone_id,obs.candidate_id
        if obs.evidence_id in self.seen_local[i]:
            return
        self.seen_local[i].add(obs.evidence_id)
        track = self.local[i].get(candidate)
        if track is None:
            self.local[i][candidate] = LocalTrack.first(obs.estimate_xy,obs.measured_at)
        else:
            if not track.update(obs.estimate_xy,obs.measured_at):
                return
        self.support[i][candidate] = (obs.evidence_id,now)
        job = self.active(i)
        if (job is None or job.kind not in ("inspect","observe") or job.params["candidate_id"] != candidate
                or obs.measured_at <= job.created_at or obs.quality < .8 or obs.object_type != job.params.get("object_type")):
            return
        # Opportunistic views on an already committed safe continuation count;
        # actor ACK alone does not count as acquisition or service.
        if job.status == "queued":
            self.book.transition(job,"running",now,"actual local acquisition on committed flight")
        job.params.setdefault("service_started_at",obs.measured_at)
        views = self.views.setdefault(job.job_id,[])
        if not views or np.linalg.norm(np.asarray(views[0])-obs.aircraft_xy) >= 20:
            views.append(obs.aircraft_xy)
        self.service_samples.setdefault(job.job_id,[]).append(obs.measured_at)
        job.params["measurements"] = len(self.service_samples[job.job_id])
        if job.kind == "inspect" and len(views) >= 2:
            self.book.transition(job,"succeeded",now,"two actual qualified views separated by20m")
        if (job.kind == "observe" and "handoff_from_job" in job.params
                and "handoff_receiver_acquired_at" not in job.params):
            source = self.book.jobs[job.params["handoff_from_job"]]
            job.params["handoff_receiver_acquired_at"] = now
            if (source.status in ("queued","running") and self.book.active.get(source.drone_id) == source.job_id
                    and source.version == job.params["handoff_from_version"] == self.book.versions[source.drone_id]):
                self.book.transition(source,"cancelled",now,"handoff acquired actual new receiver view")
                job.params["handoff_effective_at"] = now
            else:
                job.params["handoff_source_stale_at"] = now

    def aim(self,i,now):
        job = self.active(i)
        if job is None or job.kind not in ("inspect","observe"):
            return None
        track = self.local[i].get(job.params["candidate_id"])
        return track.predict(now) if track is not None else None

    def track_age(self,i,now):
        job = self.active(i)
        track = self.local[i].get(job.params.get("candidate_id")) if job else None
        return now-track.measured_at if track is not None else math.inf

    def camera_narrow(self,i,now):
        return self.aim(i,now) is not None and self.track_age(i,now) <= 3.

    def _request(self,job,now):
        self.serial += 1
        i = job.drone_id
        estimate = None
        if job.kind in ("inspect","observe"):
            candidate = job.params["candidate_id"]
            track = self.local[i][candidate]
            evidence,available = self.support[i][candidate]
            estimate = ObservedTrack(i,track.measured_at,available,tuple(track.position),tuple(track.velocity),(evidence,))
        duration = job.params.get("duration",20.)
        if job.kind == "observe" and "service_started_at" in job.params:
            duration -= now-job.params["service_started_at"]
        region = (-300.,-100.,100.,300.)[int(job.params["region_id"].split("-")[1])] if job.kind == "survey" else None
        request = PlanRequest(self.serial,self.revision,now,now+self.capture_lead_seconds,i,self.tokens(),job.kind,max(.5,duration),
                              region,estimate,self.programs,self.homes,self.wind)
        self.requests.put_nowait(request)
        row = {f.name:getattr(request,f.name) for f in fields(request) if f.name != "programs"}
        row["track"] = asdict(request.track) if request.track is not None else None
        row.update(type="plan_request",envelope_id=self.envelope_id,job_id=job.job_id)
        self.audit.append(row)
        self.pending = (request,job.job_id)
        self.pending_wall_deadline = time.monotonic()+5.

    def _altitude_interval(self,i,now):
        old,new,stamp = self.altitude_transitions[i]
        duration = abs(new-old)/self.vertical_command_limit+20.
        return (max(60.,min(old,new)-2),min(100.,max(old,new)+2)) if now-stamp < duration else (new-2,new+2)

    def _set_altitude(self,i,target,now):
        if target != self.work_altitudes[i]:
            self.altitude_transitions[i] = (self.work_altitudes[i],target,now)
            self.work_altitudes[i] = target

    def audit_tracking(self,stamp,positions):
        errors = [float(np.linalg.norm(np.asarray(pos)[:2]-p.at(stamp)[:2])) for pos,p in zip(positions,self.programs)]
        self.tracking["physics_samples"] += 1
        self.tracking["max_horizontal_error_m"] = max(self.tracking["max_horizontal_error_m"],max(errors))
        self.tracking["horizontal_tube_breach_steps"] += int(max(errors) > 6)
        altitude_bad = False
        for i,pos in enumerate(positions):
            low,high = self._altitude_interval(i,stamp)
            altitude_bad |= not low <= pos[2] <= high
            old,new,start = self.altitude_transitions[i]
            if stamp-start >= abs(new-old)/self.vertical_command_limit+20:
                self.tracking["max_settled_altitude_error_m"] = max(self.tracking["max_settled_altitude_error_m"],abs(pos[2]-self.work_altitudes[i]))
        self.tracking["altitude_envelope_breach_steps"] += int(altitude_bad)

    def _capture(self,i,now,navigation):
        sample = self.programs[i].at(now)
        state = navigation[i]
        low,high = self._altitude_interval(i,now)
        ground = sample[3]*np.array([math.cos(sample[2]),math.sin(sample[2])])
        va = float(np.linalg.norm(ground-self.wind))
        heading = air_heading_for_course(float(sample[2]),np.asarray(self.wind),va)
        metrics = {"xy_error_m":float(np.linalg.norm(state[3,:2]-sample[:2])),
                   "z_m":float(state[3,2]),"z_interval_m":[low,high],
                   "heading_error_degrees":math.degrees(abs(wrap_angle(state[1,2]-heading))),
                   "ground_speed_error_m_s":abs(float(np.linalg.norm(state[2]))-float(sample[3])),
                   "bank_error_degrees":math.degrees(abs(state[1,0]-sample[4]))}
        good = (metrics["xy_error_m"] <= 6 and low <= metrics["z_m"] <= high
                and metrics["heading_error_degrees"] <= 10 and metrics["ground_speed_error_m_s"] <= 3
                and metrics["bank_error_degrees"] <= 10)
        return good,metrics

    def _poll(self,now):
        try:
            payload = self.results.get_nowait()
        except queue.Empty:
            return
        if not isinstance(payload,bytes) or len(payload) > 1100000:
            self._planner_fault("invalid bounded planner message",now)
            return
        try:
            proposal,wall_seconds = pickle.loads(payload)
        except (pickle.PickleError,EOFError,ValueError,TypeError,AttributeError):
            self._planner_fault("malformed internal planner response",now)
            return
        if not isinstance(proposal,PlanProposal):
            self._planner_fault("wrong internal planner response type",now)
            return
        if self.pending is None or proposal.serial != self.pending[0].serial:
            if proposal.serial in self.expired_serials or self.infrastructure_errors:
                self.audit.append({"type":"late_planner_response_discarded","serial":proposal.serial,"at":now})
            else:
                self._planner_fault("planner response does not match outstanding request",now)
            return
        request,job_id = self.pending
        self.pending = None
        self.pending_wall_deadline = None
        job = self.book.jobs[job_id]
        valid = (now <= proposal.commit_at+1e-8 and proposal.revision == self.revision
                 and proposal.tokens == self.tokens() and job.status in ("queued","running"))
        self.audit.append({"type":"plan_response","serial":proposal.serial,"received_at":now,
                           "planner_wall_seconds":wall_seconds,"fresh":valid,"reason":proposal.reason,
                           "program_sha256":program_hash(proposal.program) if proposal.program else None,
                           "geometry":proposal.geometry})
        if valid and proposal.program is not None:
            self.ready = (proposal,job_id)
        else:
            self.retry_after[job_id] = now+1.

    def _planner_fault(self,reason,now):
        error = {"type":"planner_protocol_error","at":now,"reason":reason}
        self.infrastructure_errors.append(error)
        self.audit.append(error)
        self.pending,self.ready,self.pending_wall_deadline = None,None,None

    def advance(self,now,navigation,exit_dwell):
        if self.process is not None and self.process.exitcode is not None and not self.infrastructure_errors:
            self.infrastructure_errors.append({"type":"planner_process_failed","at":now,"exitcode":self.process.exitcode})
            self.audit.extend(self.infrastructure_errors)
        self._poll(now)
        if self.pending is not None:
            request,job_id = self.pending
            if now > request.commit_at and request.serial not in self.expired_serials:
                self.expired_serials.add(request.serial)
                self.audit.append({"type":"planner_capture_expired","serial":request.serial,"at":now})
            if self.pending_wall_deadline is not None and time.monotonic() > self.pending_wall_deadline:
                error = {"type":"planner_no_reply_timeout","serial":request.serial,"at":now,"wall_budget_seconds":5}
                self.infrastructure_errors.append(error)
                self.audit.append(error)
                if self.process is not None and self.process.is_alive():
                    self.process.terminate()  # no join/wait in the physical tick
                self.pending,self.pending_wall_deadline = None,None
        if self.ready is not None:
            proposal,job_id = self.ready
            if (proposal.revision != self.revision or proposal.tokens != self.tokens()
                    or self.book.jobs[job_id].status not in ("queued","running")):
                self.audit.append({"type":"proposal_invalidated","serial":proposal.serial,"at":now})
                self.ready = None
            elif now >= proposal.commit_at-1e-8:
                i = proposal.drone
                capture,metrics = self._capture(i,now,navigation)
                if not capture:
                    self.audit.append({"type":"capture_envelope_rejected","at":now,"drone":i,"serial":proposal.serial,"metrics":metrics})
                    self.retry_after[job_id] = now+1
                else:
                    programs = list(self.programs)
                    programs[i] = proposal.program
                    self.programs = tuple(programs)
                    self._set_altitude(i,96. if self.book.jobs[job_id].kind in ("inspect","observe") else self.home_altitudes[i],now)
                    self.revision += 1
                    job = self.book.jobs[job_id]
                    if job.status == "queued":
                        self.book.transition(job,"running",now,"physical program adopted")
                    job.params["planned_service_from"] = proposal.service_start
                    job.params["planned_service_until"] = proposal.service_end
                    self.refreshed_at[job_id] = now
                    if job.kind == "survey":
                        center = (-300.,-100.,100.,300.)[int(job.params["region_id"].split("-")[1])]
                        self.survey_monitors[job_id] = SurveyRoute(center)
                    self.audit.append({"type":"program_committed","serial":proposal.serial,"at":now,
                                       "revision":self.revision,"drone":i,"job_id":job_id,
                                       "program_sha256":program_hash(proposal.program)})
                self.ready = None
        for i in range(4):
            job = self.active(i)
            if job is None:
                continue
            if job.kind == "exit" and exit_dwell[i] >= 20-1e-6:
                if job.status == "queued":
                    self.book.transition(job,"running",now,"already physically in home exit")
                self.book.transition(job,"succeeded",now,"actual home-circle20s dwell")
            elif job.kind == "survey" and job.status == "running":
                start = job.params["planned_service_from"]
                if now >= start:
                    monitor = self.survey_monitors[job.job_id]
                    monitor.target(navigation[i][3],self.home_altitudes[i])
                    if "service_started_at" not in job.params and monitor.cross_track_error(navigation[i][3]) <= 6:
                        job.params["service_started_at"] = now
                    if "service_started_at" in job.params and now-job.params["service_started_at"] >= job.params["duration"]:
                        self.book.transition(job,"succeeded" if monitor.laps >= 1 else "failed",now,
                                             "actual pass and service interval" if monitor.laps >= 1 else "no complete physical pass")
            elif job.kind in ("inspect","observe"):
                if job.kind == "inspect" and now-job.created_at >= 90:
                    self.book.transition(job,"failed",now,"no qualified second view within90s")
                elif "service_started_at" not in job.params and now-job.created_at >= 90:
                    self.book.transition(job,"failed",now,"no qualified acquisition within90s")
                elif job.kind == "observe" and "service_started_at" in job.params:
                    start,duration = job.params["service_started_at"],job.params["duration"]
                    if now-start >= duration:
                        samples = self.service_samples.get(job.job_id,[])
                        ends = np.r_[np.arange(3,duration,3),duration]
                        complete = all(any(start+a < s <= start+b for s in samples) for a,b in zip(np.r_[0,ends[:-1]],ends))
                        self.book.transition(job,"succeeded" if complete else "failed",now,
                                             "local measurement bins complete" if complete else "local observation gaps retained")
            if job.status == "queued" and now-job.created_at > 90:
                self.book.transition(job,"failed",now,"finite acquisition/traffic queue timeout")
        if self.process is not None and not self.infrastructure_errors and self.pending is None and self.ready is None:
            choices = []
            for i in range(4):
                job = self.active(i)
                if job is None or self.retry_after.get(job.job_id,0) > now:
                    continue
                if (job.status == "queued" or (job.kind == "inspect" and job.job_id not in self.refreshed_at)
                        or (job.kind == "observe" and now-self.refreshed_at.get(job.job_id,-100) >= 1
                            and (job.job_id not in self.refreshed_at
                                 or now >= job.params.get("planned_service_from",now)))):
                    choices.append((0 if job.kind == "exit" else (1 if job.status == "queued" else 2),
                                    self.refreshed_at.get(job.job_id,job.created_at),i,job))
            if choices:
                try:
                    self._request(min(choices,key=lambda row:row[:3])[-1],now)
                except queue.Full:
                    pass
        targets,exits,modes = [],[],[]
        for i,p in enumerate(self.programs):
            exiting = now >= p.prefix.duration
            job = self.active(i)
            if exiting:
                self._set_altitude(i,self.home_altitudes[i],now)
            altitude = self.work_altitudes[i]
            targets.append(p.target(now,navigation[i][3],altitude))
            exits.append(exiting)
            modes.append("exit" if exiting else (job.kind if job else "continuation"))
        return targets,exits,modes

    def close(self):
        if self.process is not None:
            try:
                self.requests.put_nowait(None)
            except queue.Full:
                pass
            self.process.join(timeout=.2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2.)
        for q in (self.requests,self.results):
            q.cancel_join_thread()
            q.close()
