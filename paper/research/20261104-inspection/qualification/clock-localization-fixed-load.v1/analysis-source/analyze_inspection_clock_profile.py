"""Recount a completed engineering clock profile; grants no qualification."""
from array import array
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recount(directory):
    directory = Path(directory)
    metadata = json.loads((directory/'clock-profile.json').read_text())
    if metadata['byte_order'] != sys.byteorder or metadata['dtype'] != 'native_float64':
        raise ValueError('decode the declared native float representation explicitly')
    inputs = {}
    for name, expected in metadata['files'].items():
        path = directory/name
        if sha(path) != expected['sha256'] or path.stat().st_size != expected['bytes']:
            raise ValueError('profile binary differs from its completed metadata')
        inputs[name] = sha(path)
    def decode(name, fields, count):
        values = array('d'); values.frombytes((directory/name).read_bytes())
        if len(values) != len(fields)*count or not all(math.isfinite(v) for v in values):
            raise ValueError('complete finite rows required')
        return [dict(zip(fields, values[i:i+len(fields)])) for i in range(0,len(values),len(fields))]
    rows = decode('clock-ticks.f64', metadata['fields'], metadata['rows'])
    events = decode('clock-gc-events.f64', metadata['gc_fields'], metadata['gc_rows'])
    if len(rows) != 72000 or any(abs(row['sim_after']-(i+1)/120)>1e-9 for i,row in enumerate(rows)):
        raise ValueError('one complete600s/120Hz denominator required')
    lags = [row['lag'] for row in rows]
    statistics = {'max_lag':max(lags), 'p95_lag':sorted(lags)[int(.95*(len(lags)-1))],
                  'ticks_over_0p1':sum(v>.1 for v in lags)}
    if any(metadata[k] != v for k,v in statistics.items()):
        raise ValueError('raw lag recount differs from metadata')
    world = json.loads((directory/'world.json').read_text())
    for key, value in [('max_clock_lag_seconds',statistics['max_lag']),
                       ('clock_lag_p95_seconds',statistics['p95_lag']),
                       ('clock_lag_over_0p1_seconds_ticks',statistics['ticks_over_0p1'])]:
        if world[key] != value: raise ValueError('world lag differs from full array')
    stages = ('begin','wake','rpc_end','step_end','progress_end','stats_end','pub_end','log_end')
    intervals = {}
    for before, after in zip(stages,stages[1:]):
        measurements = []
        for i,row in enumerate(rows):
            value = {'tick_index':i, 'sim_after':row['sim_after'], 'lag':row['lag'],
                     'sleep_requested':row['sleep_requested'],
                     'wall_start':row[before+'_wall'], 'wall_end':row[after+'_wall']}
            value.update({clock:row[after+'_'+clock]-row[before+'_'+clock]
                          for clock in ('wall','thread_cpu','process_cpu')})
            if any(value[k]<0 for k in ('wall','thread_cpu','process_cpu')):
                raise ValueError('nonmonotonic stage clock')
            measurements.append(value)
        intervals[before+'__'+after] = sorted(measurements,key=lambda r:r['wall'],reverse=True)[:10]
    peak = max(range(len(rows)),key=lambda i:rows[i]['lag'])
    stack, pairs = {}, []
    for event in events:
        key = (event['thread_id'],event['generation'])
        if event['event']==1:
            if key in stack: raise ValueError('duplicate unclosed GC start')
            stack[key] = event
        elif event['event']==2:
            if key not in stack: raise ValueError('GC stop without retained start')
            start = stack.pop(key)
            pairs.append({'generation':key[1], 'main_thread':key[0]==metadata['main_thread_id'],
                'start':start,'end':event,
                **{clock:event[clock]-start[clock] for clock in ('wall','thread_cpu','process_cpu')}})
        else: raise ValueError('unknown GC event')
    if stack or metadata['gc_overflow']:
        raise ValueError('cannot claim a complete GC event pairing')
    previous = rows[peak-1] if peak else None
    overlap = []
    if previous:
        left,right = previous['pub_end_wall'],previous['log_end_wall']
        overlap = [pair for pair in pairs if pair['start']['wall']<=right and pair['end']['wall']>=left]
    inputs.update({n:sha(directory/n) for n in ('clock-profile.json','world.json','diagnostic.json')})
    return {'role':'full raw engineering profile recount; not original-failure attribution or qualification',
        'analysis_source_sha256':sha(__file__), 'inputs':inputs,
        'statistics':statistics, 'rows':len(rows), 'gc_pairs':len(pairs), 'gc_overflow':metadata['gc_overflow'],
        'peak_tick_index':peak,'peak_tick':rows[peak],'previous_tick':previous,
        'previous_tick_is_1Hz_logging_tick':bool(previous and round(previous['sim_after']*240)%240==0),
        'gc_overlapping_previous_pub_end_to_log_end':overlap,
        'longest_phase_intervals':intervals,
        'longest_gc_intervals':sorted(pairs,key=lambda r:r['wall'],reverse=True)[:10],
        'limits':['Stage intervals include clock-marker overhead and possible off-CPU waiting.',
                  'A phase name does not establish that its conditional operation executed.',
                  'Low process CPU relative to wall time does not identify a unique OS/IPC/GIL cause.',
                  'B0 with fixed prompts differs from the original adaptive B4; no reproduced invalidity or repair claim.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();result=recount(args.directory)
    with args.out.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'statistics':result['statistics'],'peak_tick':result['peak_tick_index'],
                      'previous_tick_is_1Hz_logging_tick':result['previous_tick_is_1Hz_logging_tick'],
                      'gc_pairs':result['gc_pairs']}))
