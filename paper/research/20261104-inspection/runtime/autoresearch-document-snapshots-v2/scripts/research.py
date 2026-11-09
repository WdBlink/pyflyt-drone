#!/usr/bin/env python3
"""Research-to-paper controller. See --help and references/runtime-contract.md."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import time
import subprocess
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.runtime import Controller, Refusal, read_json


def input_json(root, name):
    """Accept run-relative inputs from any cwd; refuse ambiguous legacy paths."""
    path = Path(name)
    if not path.is_absolute():
        relative = root / path
        if relative.is_file() and path.is_file() and relative.resolve() != path.resolve():
            raise Refusal('Ambiguous JSON input; use an absolute path: ' + name)
        path = relative if relative.is_file() or not path.is_file() else path
    return read_json(path)


def trace(root, command, result, code):
    # Transport output must not alter prepare-round's research-document digest.
    directory = root.parent / (root.name + '-cli')
    try:
        if directory.is_symlink():
            raise OSError('Trace directory cannot be a symlink')
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (str(time.time_ns()) + '-' + command + '.json')
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as stream:
            json.dump({'command': command, 'exit_code': code, 'response': result}, stream, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
    except OSError as error:
        print('CLI trace unavailable: ' + str(error), file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='New-format run directory, never a legacy active run')
    sub = parser.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init'); init.add_argument('--brief', required=True)
    approve = sub.add_parser('approve'); approve.add_argument('--brief-sha256', required=True); approve.add_argument('--actor', required=True)
    approve.add_argument('--authorization', help='Optional evidence JSON for a user-delegated research decision')
    context = sub.add_parser('context'); context.add_argument('--writing-case', action='append', default=[]); sub.add_parser('status')
    preview = sub.add_parser('draft-preview'); preview.add_argument('--pdf', required=True); preview.add_argument('--out', required=True)
    tool = sub.add_parser('tool'); tool.add_argument('--name', required=True); tool.add_argument('--packet', required=True); tool.add_argument('--out'); tool.add_argument('--publish', action='store_true')
    artifact = sub.add_parser('artifact', help='State-free, versioned production tools, including historical input reuse')
    artifact.add_argument('--name', required=True); artifact.add_argument('--packet', required=True); artifact.add_argument('--out')
    review = sub.add_parser('prepare-review'); review.add_argument('--out', required=True)
    assembly = sub.add_parser('assemble-review'); assembly.add_argument('--packet', required=True)
    begin = sub.add_parser('begin-round'); begin.add_argument('--session-id')
    ask = sub.add_parser('ask-user'); ask.add_argument('--kind', choices=['venue','budget'], required=True); ask.add_argument('--packet', required=True)
    budget_answer = sub.add_parser('confirm-budget'); budget_answer.add_argument('--response', required=True); budget_answer.add_argument('--operation-id', required=True)
    reserve = sub.add_parser('budget-reserve'); reserve.add_argument('--packet', required=True); reserve.add_argument('--operation-id', required=True)
    settle = sub.add_parser('budget-settle'); settle.add_argument('--reservation'); settle.add_argument('--packet', required=True); settle.add_argument('--operation-id', required=True)
    resume = sub.add_parser('resume-task'); resume.add_argument('--response', required=True); resume.add_argument('--operation-id', required=True)
    sc = sub.add_parser('set-contract'); sc.add_argument('--brief', required=True); sc.add_argument('--operation-id', required=True)
    cv = sub.add_parser('confirm-venue'); cv.add_argument('--packet', required=True); cv.add_argument('--actor', required=True); cv.add_argument('--comparison-sha256', required=True); cv.add_argument('--operation-id', required=True)
    cv.add_argument('--authorization', help='Removed: delegation does not answer the venue question; use --response')
    cv.add_argument('--response', required=True, help='Source-bound actual answer to the issued venue question')
    recover = sub.add_parser('recover'); recover.add_argument('--manifest')
    migrate = sub.add_parser('migrate-state-reader'); migrate.add_argument('--packet', required=True); migrate.add_argument('--operation-id', required=True)
    md = sub.add_parser('migrate-round-documents'); md.add_argument('--packet', required=True); md.add_argument('--operation-id', required=True)
    rd = sub.add_parser('read-material'); rd.add_argument('--path', required=True); rd.add_argument('--start', type=int, default=0); rd.add_argument('--end', type=int)
    prep = sub.add_parser('prepare-round'); prep.add_argument('--evidence', nargs='*', action='extend', default=[]); prep.add_argument('--current-plan')
    rnd = sub.add_parser('apply-round'); rnd.add_argument('--request'); rnd.add_argument('--decision', required=True); rnd.add_argument('--operation-id', required=True)
    revise = sub.add_parser('revise-orders'); revise.add_argument('--packet', required=True); revise.add_argument('--reason', required=True); revise.add_argument('--operation-id', required=True)
    for name in ('commit', 'bind', 'bind-all'):
        item = sub.add_parser(name); item.add_argument('--packet', required=True); item.add_argument('--operation-id', required=True)
    execute = sub.add_parser('execute'); execute.add_argument('--work-id', required=True); execute.add_argument('--operation-id', required=True)
    rewind = sub.add_parser('rewind'); rewind.add_argument('--stage', required=True); rewind.add_argument('--reason', required=True); rewind.add_argument('--operation-id', required=True)
    rewind.add_argument('--preserve', nargs='*', default=[], help='Copy accepted affected artifacts before invalidating their receipts')
    reconcile = sub.add_parser('reconcile'); reconcile.add_argument('--attempt', required=True); reconcile.add_argument('--outcome', required=True); reconcile.add_argument('--evidence', required=True); reconcile.add_argument('--operation-id', required=True)
    final = sub.add_parser('finalize'); final.add_argument('--operation-id', required=True)
    delivery = sub.add_parser('deliver'); delivery.add_argument('--decision', required=True); delivery.add_argument('--operation-id', required=True)
    verify = sub.add_parser('verify-export'); verify.add_argument('--manifest', required=True)
    for name in ('fork-trigger', 'fork-plan'):
        fork = sub.add_parser(name); fork.add_argument('--config', required=True)
    create = sub.add_parser('fork-create'); create.add_argument('--repo', required=True); create.add_argument('--plan', required=True); create.add_argument('--destination', required=True)
    select = sub.add_parser('fork-select'); select.add_argument('--plan', required=True); select.add_argument('--candidates', required=True); select.add_argument('--incumbent-metric', type=float, required=True); select.add_argument('--promote-repo')
    args = parser.parse_args()
    ctl = Controller(args.run)
    read = lambda name: input_json(ctl.root, name)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            if args.command.startswith('fork-'):
                from lib import exploration
                if args.command == 'fork-trigger': result = exploration.trigger(**read(args.config))
                elif args.command == 'fork-plan': result = exploration.request(**read(args.config))
                elif args.command == 'fork-create':
                    repo = Path(args.repo).resolve()
                    if ctl.root == repo or ctl.root.is_relative_to(repo):
                        raise Refusal('Keep the authoritative paper run outside an exploration checkout')
                    result = exploration.create_worktrees(repo, read(args.plan), args.destination)
                elif args.promote_repo:
                    repo = Path(args.promote_repo).resolve()
                    if ctl.root == repo or ctl.root.is_relative_to(repo):
                        raise Refusal('Keep paper state outside the exploration checkout')
                    result = exploration.promote(repo, read(args.plan), read(args.candidates), args.incumbent_metric)
                else: result = exploration.select(read(args.plan), read(args.candidates), args.incumbent_metric)
            elif args.command == 'artifact':
                from lib import production, reviewing, experiments
                packet = read(args.packet)
                if args.name == 'review-request': result = reviewing.prepare(packet, ctl.root)
                elif args.name == 'review-assemble': result = reviewing.assemble(packet, ctl.root)
                elif args.name == 'experiment-template': result = experiments.template(packet, ctl.root, ctl.package)
                else: result = production.run(args.name, packet, ctl.root)
                if args.out:
                    production.write_once(ctl.root, args.out, production.json_bytes(result))
            elif args.command == 'prepare-review': result = ctl.prepare_review(args.out)
            elif args.command == 'assemble-review': result = ctl.assemble_review(read(args.packet))
            elif args.command == 'init': result = ctl.init(read(args.brief))
            elif args.command == 'approve': result = ctl.approve(args.brief_sha256, args.actor, read(args.authorization) if args.authorization else None)
            elif args.command == 'context': result = ctl.context(args.writing_case)
            elif args.command == 'draft-preview': result = ctl.draft_preview(args.pdf, args.out)
            elif args.command == 'tool': result = ctl.tool(args.name, read(args.packet), args.out, publish=args.publish)
            elif args.command == 'begin-round': result = ctl.begin_round(session_id=args.session_id)
            elif args.command == 'ask-user': result = ctl.prepare_user_input(args.kind, read(args.packet))
            elif args.command == 'confirm-budget': result = ctl.confirm_budget(read(args.response), args.operation_id)
            elif args.command == 'budget-reserve': result = ctl.reserve_budget(read(args.packet), args.operation_id)
            elif args.command == 'budget-settle': result = ctl.settle_budget(args.reservation, read(args.packet), args.operation_id)
            elif args.command == 'resume-task': result = ctl.resume_task(read(args.response), args.operation_id)
            elif args.command == 'set-contract': result = ctl.set_contract(read(args.brief), args.operation_id)
            elif args.command == 'confirm-venue': result = ctl.confirm_venue(read(args.packet), args.actor, args.comparison_sha256, args.operation_id, read(args.authorization) if args.authorization else None, read(args.response) if args.response else None)
            elif args.command == 'migrate-round-documents': result = ctl.migrate_round_documents(read(args.packet), args.operation_id)
            elif args.command == 'recover': result = ctl.recover(args.manifest)
            elif args.command == 'migrate-state-reader': result = ctl.migrate_state_reader(read(args.packet), args.operation_id)
            elif args.command == 'read-material': result = ctl.read_material(args.path, args.start, args.end)
            elif args.command == 'prepare-round': result = ctl.prepare_round(args.evidence, args.current_plan)
            elif args.command == 'apply-round': result = ctl.apply_round(read(args.request) if args.request else None, read(args.decision), args.operation_id)
            elif args.command == 'status':
                from lib.budget import summary
                from lib.rounds import checkpoint
                state = ctl.load(); result = {'stage': len(state['receipts']), 'stale': ctl.stale(state), 'approval': state['approval'], 'attempts': state['attempts'], 'delivery': state['delivery'], 'budget': summary(state), 'checkpoint': checkpoint(state), 'user_questions': [q['question'] for q in state['user_questions'].values() if q['status'] == 'needs_input']}
            elif args.command == 'commit': result = ctl.commit(read(args.packet), args.operation_id)
            elif args.command == 'revise-orders': result = ctl.revise_orders(read(args.packet), args.reason, args.operation_id)
            elif args.command == 'bind': result = ctl.bind(read(args.packet), args.operation_id)
            elif args.command == 'bind-all': result = ctl.bind_all(read(args.packet), args.operation_id)
            elif args.command == 'execute': result = ctl.execute(args.work_id, args.operation_id)
            elif args.command == 'rewind': result = ctl.rewind(args.stage, args.reason, args.operation_id, args.preserve)
            elif args.command == 'reconcile': result = ctl.reconcile(args.attempt, args.outcome, args.evidence, args.operation_id)
            elif args.command == 'finalize': result = ctl.finalize(args.operation_id)
            elif args.command == 'deliver': result = ctl.deliver(read(args.decision), args.operation_id)
            else: result = ctl.verify_export(args.manifest)
        trace(ctl.root, args.command, result, 0)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError, zipfile.BadZipFile) as exc:
        result = {'status': 'refused', 'reason': str(exc)}
        trace(ctl.root, args.command, result, 2)
        print(json.dumps(result, ensure_ascii=False))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
