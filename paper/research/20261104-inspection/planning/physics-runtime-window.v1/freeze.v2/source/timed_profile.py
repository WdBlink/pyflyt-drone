"""Engineering-only runtime-window integration; original tick loop unchanged."""
import gc
import hashlib
import json
from pathlib import Path

from eval.inspection_clock_profile_rusage import RusageClockProfile
from physics_runtime import PhysicsRuntimeWindow


class TimedClockProfile(RusageClockProfile):
    def __init__(self,duration):
        super().__init__(duration)
        try:
            self.window=PhysicsRuntimeWindow()
            self.window.start()
            self.configuration={'enabled':gc.isenabled(),'threshold':gc.get_threshold(),
                'count':gc.get_count(),'frozen_count':gc.get_freeze_count()}
        except BaseException:
            self.stop();raise

    def stop(self):
        super().stop()
        if hasattr(self,'window'):self.window.close()

    def save(self,directory):
        value=super().save(directory)
        directory=Path(directory)
        value['runtime_policy']=self.window.audit
        value['recorder_changes_gc_configuration']=True
        value['limits'][-1]='Unchanged epoch/pacer/threshold; explicit physics GC deferral and self-thread soft real-time policy, restored outside control.'
        path=directory/'clock-profile.json'
        # The existing recorder writes once; only this new variant owns its
        # completed metadata and updates the adjacent cross-file hashes.
        path.write_text(json.dumps({key:item for key,item in value.items() if key!='rusage_profile'},indent=2)+'\n')
        usage_path=directory/'clock-rusage-profile.json'
        usage=json.loads(usage_path.read_text())
        usage['base_clock_profile_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        usage['runtime_policy']=self.window.audit
        usage['requested_qos_before_epoch']['scope']+='; queried before applying the separately recorded Mach constraint'
        usage['limits'][0]='Explicit worker-only runtime-window changes are recorded; epoch/pacer/lag threshold unchanged.'
        usage_path.write_text(json.dumps(usage,indent=2)+'\n')
        value['rusage_profile']['sha256']=hashlib.sha256(usage_path.read_bytes()).hexdigest()
        return value
