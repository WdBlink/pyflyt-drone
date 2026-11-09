#!/usr/bin/env python3
"""Run offline empirical or source-synthesis fixtures. No model, network or real venue claim."""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.runtime import Controller, Refusal, digest, file_hash


def put(root, name, value):
    path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n' if not isinstance(value, str) else value)
    return name


def packet(stage, files, **fields):
    return {'stage': stage, 'producer': 'fixture-author', 'artifacts': list(dict.fromkeys(files)), **fields}


def answer_question(ctl, kind, packet, answer, name):
    question = ctl.prepare_user_input(kind, packet)['question']
    quote = 'SOFTWARE FIXTURE scripted answer, not a human response: ' + json.dumps(answer, sort_keys=True)
    path = put(ctl.root, 'fixture-answers/' + name + '.txt', quote)
    return {'question_sha256': question['question_sha256'], 'answer': answer,
            'human_response': {'path': path, 'sha256': file_hash(ctl.root / path), 'quote': quote}}


def make_brief(survey=False):
    brief = {'software_fixture': True, 'question': 'Exercise a paper workflow using fixed local CPU transformations; no scientific claim',
            'venue': 'Synthetic Software Fixture', 'direction': 'software-contract testing', 'paper_type': 'original',
            'scope': 'Local software validation only', 'angle': 'Evidence and artifact contract closure', 'audience': 'Developers reviewing the Skill',
            'resources': {'authorization': 'offline software verification only', 'cpu': 'sequential local fixture'},
            'budget': {'calls': 0},
            'reviewers': ['fixture-reviewer'],
            'deliverable_plan': [{'id': key, 'kind': kind, 'title': title,
                                  'purpose': 'Verify ' + title + ' against the measured software fixture',
                                  'claim': 'C1', 'acceptance_criteria': ['Correctly represent the complete planned fixture outcomes'],
                                  'evidence_requirements': ['Frozen design, raw logs, numerical trace and manuscript location']}
                                 for key, kind, title in [('F1', 'figure', 'Paired fixture errors'), ('T1', 'table', 'Fixture statistics'),
                                                          ('S1', 'section', 'Results and limitations'), ('M1', 'manuscript', 'Compiled software fixture')]]}
    if survey:
        brief.update(question='Exercise source-bounded synthesis of a synthetic record; no scientific claim', paper_type='survey',
                     not_applicable={'experiment': 'Approved non-empirical synthesis; no new measurement or comparative empirical claim.'})
        for item in brief['deliverable_plan']:
            item.update(claim='LC1', evidence_requirements=['Verified source spans and manuscript location'])
        brief['deliverable_plan'][0]['title'] = 'Source evidence diagram'
        brief['deliverable_plan'][1]['title'] = 'Source comparison table'
    return brief


EVALUATOR = '''import json
runs=[]
for index,case in enumerate('abcd',1):
 for seed in (0,1):
  for condition in ('baseline','candidate'):
   status='failed' if condition=='candidate' and case=='c' else 'stopped' if condition=='candidate' and case=='d' else 'completed'
   value=(index+seed*.2+(1 if condition=='candidate' else 0)) if status=='completed' else None
   runs.append(dict(case_id=case,seed=seed,condition=condition,status=status,value=value))
print(json.dumps(runs,allow_nan=False))
'''


SAMPLE_TEX = r'''\documentclass{article}
\usepackage[labelsep=period]{caption}
\title{Synthetic software venue sample}
\begin{document}\maketitle
This sample is generated for software validation. It is not an accepted research paper.
Figure 1 and Table 1 show the fixture layout.
\begin{figure}[h]\centering\fbox{Synthetic layout}\caption{Sample quantitative results.}\end{figure}
\begin{table}[h]\centering\begin{tabular}{lr}Case&Value\\A&1\\\end{tabular}\caption{Sample main results.}\end{table}
\end{document}
'''


class FixturePaused(Exception):
    pass


def run(root, *, stop_after=None, undecided=False, survey=False, budget=None):
    from lib import planning, science
    root = Path(root).resolve()
    if (root / '.paper/control.json').exists():
        raise Refusal('Fixture requires a fresh run; it never resets a historical run')
    root.mkdir(parents=True, exist_ok=True)
    ctl = Controller(root)
    brief = make_brief(survey)
    if budget is not None:
        brief['budget'] = budget
    initial = copy.deepcopy(brief)
    if undecided: initial.update(venue='undecided', deliverable_plan=[], contract_status='draft')
    result = ctl.init(initial)
    put(root, 'brief.json', initial)
    if not undecided: ctl.approve(result['brief_sha256'], 'offline software fixture')
    contexts = []
    def commit(p):
        context = ctl.context(writing_cases=['E07'] if p['stage'] == 'write' else [])
        if p['stage'] == 'write':
            assert context['writing_guidance']['selected_cases'][0]['id'] == 'E07'
            assert context['writing_guidance']['principles']
        assert context['stage'] == p['stage'] and context['knowledge'] and context['upstream_protocol']
        contexts.append({'stage': p['stage'], 'knowledge_sha256': context['knowledge_sha256']})
        put(root, p['stage'] + '/packet.json', p)
        receipt = ctl.commit(p, 'commit-' + p['stage'])
        if p['stage'] == stop_after:
            raise FixturePaused(p['stage'])
        return receipt

    text = ('Offline source-synthesis fixture: classify one synthetic source, retaining its exact scope. No new measurements are authorized.' if survey else
            'Offline fixture: compare two fixed transformations on four cases, preserving failures and stops. Only local sequential CPU work is authorized.')
    material = put(root, 'materials/project.md', text)
    evidence = {'material_id': 'B1', 'start': 0, 'end': len(text), 'quote': text}
    intake = packet('intake', [material],
                    materials=[{'id': 'B1', 'path': material, 'sha256': file_hash(root / material), 'availability': 'readable', 'read_spans': [[0, len(text)]]}],
                    evidence=[evidence],
                    feasibility={'status': 'feasible', 'selected_option': 'fixed-cpu',
                                 'options': [{'id': 'fixed-cpu', 'implementation': 'Known fixed CPU fixture evaluator', 'selection_reason': 'No external effect', 'evidence': [evidence]},
                                             {'id': 'live-model', 'implementation': 'Paid remote model', 'selection_reason': 'Excluded by offline authorization', 'evidence': [evidence]}],
                                 'resources': {'authorization': 'offline fixture only', 'cost_basis': 'zero paid model calls; local CPU'},
                                 'experiments': [{'claim': 'C1', 'metric': 'error', 'comparison': 'baseline vs candidate', 'protocol': 'paired cases and seeds', 'completion': 'all planned attempts', 'failure_route': 'retain outcomes and frozen penalty'}]})
    if survey:
        intake['feasibility'].update(experiments=[], no_experiment_reason=brief['not_applicable']['experiment'])
        intake['feasibility']['options'][0]['implementation'] = 'Source-bounded synthesis of a local synthetic record'
    ctl.read_material(material)
    commit(intake)

    rules = put(root, 'venue/rules.md', 'SOFTWARE FIXTURE official-rule stand-in. Eight pages excluding references and appendix. One article, one results plot and one table; no real acceptance claim.')
    sample_tex = SAMPLE_TEX.replace('quantitative results', 'method overview').replace('main results', 'taxonomy') if survey else SAMPLE_TEX
    template = put(root, 'venue/template.tex', sample_tex)
    sample_source = put(root, 'venue/sample.tex', sample_tex)
    ctl._action(ctl.load(), {'type': 'compile_tex', 'source': sample_source, 'pdf': 'venue/sample.pdf'})
    accepted = put(root, 'venue/acceptance.md', 'Synthetic software fixture acceptance stand-in, not a real accepted paper.')
    plan_items = [dict(id='F1', kind='figure', type='quantitative_results', question='How do the measured cases compare?', claim='C1', data='experiment/results.json', section='Results', priority='must'),
                  dict(id='T1', kind='table', type='main_results', question='What are the full-denominator statistics?', claim='C1', data='experiment/results.json', section='Results', priority='must')]
    if survey:
        for item, kind in zip(plan_items, ('method_overview', 'taxonomy')):
            item.update(type=kind, question='Which unit and aggregation does the verified source describe?', claim='LC1', data='literature/source.txt')
    plan_name = put(root, 'plan/figure_table_plan.json', {'items': plan_items, 'deviation_reason': ''})
    venue = packet('venue', [rules, template, sample_source, 'venue/sample.pdf', accepted, plan_name,
                             'venue/measurements.json', 'venue/profile.json', 'venue/check.json'],
                   software_fixture=True, venue=brief['venue'], direction=brief['direction'], paper_type=brief['paper_type'],
                   official_rules={'path': rules, 'url': 'https://example.invalid/software-fixture/rules', 'sha256': file_hash(root / rules), 'basis': 'official_requirement'},
                   template={'path': template, 'url': 'https://example.invalid/software-fixture/template', 'sha256': file_hash(root / template), 'status': 'available'},
                   samples=[{'id': 'sample1', 'pdf': 'venue/sample.pdf', 'venue_label': brief['venue'], 'year': 2026,
                             'acceptance_evidence': {'path': accepted, 'url': 'https://example.invalid/software-fixture/acceptance', 'sha256': file_hash(root / accepted)},
                             'direction_match': 'Matches software fixture only',
                             'design': {'figures': [{'type': 'quantitative_results', 'placement': 'body'}], 'tables': [{'type': 'main_results', 'placement': 'body'}]}}],
                   sampling_limitations='One synthetic software sample; no claim about real venue conventions.', figure_table_plan=plan_name,
                   measurements='venue/measurements.json', venue_profile='venue/profile.json', venue_plan_check='venue/check.json')
    from lib import venue_tools
    source_rule = {**venue['official_rules'], 'quote': 'Eight pages excluding references and appendix.'}
    venue['venue_selection'] = {'mode': 'existing_target', 'selected_id': 'V1',
        'criteria': ['scope fit', 'article type', 'official format and access'], 'rationale': 'The software target is explicitly selected in the fixture Brief.',
        'candidates': [{'id': 'V1', 'venue': brief['venue'], 'kind': 'journal', 'scope_fit': 'Software verification only.',
            'article_type_fit': 'Synthetic ' + brief['paper_type'] + ' software article.', 'constraints': 'No real submission or model research.',
            'cost': {'status': 'unknown', 'basis': 'No actual journal fees investigated.'}, 'timing': {'status': 'unknown', 'basis': 'No real editorial timing investigated.'},
            'advantages': ['Exercises a sourced target.'], 'limitations': ['Not a real journal.'], 'evidence': [source_rule]}]}
    venue['sampling_policy'] = {'kind': 'journal', 'as_of': '2026-11-04', 'fixture_exception_reason': 'One generated PDF exercises the pipeline only; 8 actual PDFs are tested separately.',
                                'measurement_warning_review': 'The sample has no References heading; total-pages fallback is retained, not an official body-page measurement.'}
    venue['page_budget'] = {'target': 8, 'limit_type': 'hard_limit', 'basis': 'official_requirement', 'includes_references': False, 'includes_appendix': False,
                            'source': source_rule, 'reason': 'Synthetic author rules exercise the explicit page definition.'}
    venue['venue_comparison'] = 'venue/comparison.json'; venue['artifacts'].append('venue/comparison.json')
    designs = venue_tools.design_template(venue, root)['papers']['sample1']
    for plural, kind in [('figures', plan_items[0]['type']), ('tables', plan_items[1]['type'])]:
        for visual in designs[plural]:
            visual.update(type=kind, placement='body', question='Which fixture observations are compared?', claim='The complete software observations are shown.',
                          evidence='Synthetic local fixture data.', design_takeaway='Keep comparison conditions and units visible.', limitations='No real venue or research claim.')
    venue['samples'][0]['design'] = {k: designs[k] for k in ('figures','tables')}
    venue['samples'][0]['visual_read'] = {'pages': [1], 'method': 'rendered_pdf', 'notes': 'Scripted software reader declaration; not real venue research.'}
    if undecided:
        other = copy.deepcopy(venue['venue_selection']['candidates'][0]); other.update(id='V2', venue='Synthetic Alternative', limitations=['Different synthetic scope; not selected.'])
        venue['venue_selection']['mode'] = 'candidate_comparison'
        venue['venue_selection']['candidates'].append(other)
    comparison = ctl.tool('venue-compare', venue)['value']
    response = answer_question(ctl, 'venue', venue, {'candidate_id': 'V1'}, 'venue')
    ctl.confirm_venue(venue, 'offline fixture selection', comparison['comparison_sha256'], 'select-fixture-venue', response=response)
    venue['reading_index'] = 'venue/reading-index.md'
    venue['artifacts'].append(venue['reading_index'])
    ctl.tool('venue-calibrate', venue, publish=True)
    commit(venue)
    if undecided:
        concrete = ctl.set_contract(brief, 'set-fixture-contract')
        ctl.approve(concrete['brief_sha256'], 'offline fixture final Brief')

    if not survey:
        evaluator = put(root, 'experiment/evaluator.py', EVALUATOR)
        design = {'experiment_id': 'E1', 'claim_id': 'C1', 'hypothesis': 'Candidate reduces fixture error',
                  'independent_variable': 'transformation', 'dependent_variable': 'error', 'control_variables': ['same inputs', 'same evaluator'],
                  'metric': {'name': 'error', 'unit': 'points', 'direction': 'lower'}, 'statistical_unit': 'case',
                  'cases': list('abcd'), 'seeds': [0, 1], 'conditions': ['baseline', 'candidate'], 'failure_policy': 'penalty', 'failure_value': 10,
                  'statistics': {'estimator': 'mean', 'interval': 'sd', 'paired': True},
                  'data_roles': [{'id': 'fixture-dev', 'role': 'development', 'used_for_selection': True}, {'id': 'fixture-holdout', 'role': 'holdout', 'used_for_selection': False}],
                  'evaluator': {'path': evaluator, 'sha256': file_hash(root / evaluator)}, 'planned_at': '2026-11-03T00:00:00Z'}
        design_name = put(root, 'experiment/design.json', design)
    snapshot = put(root, 'plan/snapshot.json', planning.plan_snapshot(root / plan_name))
    orders = [dict(id='W-experiment', producer='fixture-author', requirement_ids=['F1', 'T1'], actions=[{'type': 'validate_packet', 'packet': 'experiment/packet.json'}]),
              dict(id='W-figures', producer='fixture-author', requirement_ids=['F1', 'T1'], actions=[{'type': 'render_statistics', 'input': 'experiment/results.json', 'table': 'figures/summary.tex', 'figure': 'figures/results.pdf'}]),
              dict(id='W-writing', producer='fixture-author', requirement_ids=['S1', 'M1'], actions=[{'type': 'compile_tex', 'source': 'manuscript/main.tex', 'pdf': 'manuscript/main.pdf'}])]
    if survey:
        orders = [dict(id='W-figures', producer='fixture-author', requirement_ids=['F1', 'T1'], actions=[
                      {'type': 'compile_tex', 'source': 'figures/synthesis.tex', 'pdf': 'figures/results.pdf'},
                      {'type': 'copy_artifact', 'source': 'literature/classification.tex', 'target': 'figures/summary.tex'}]), orders[-1]]
    plan = packet('plan', [plan_name, snapshot], figure_table_plan=plan_name, plan_snapshot=snapshot, work_orders=orders)
    if not survey:
        plan['artifacts'] += [design_name, evaluator]
        plan['experiment_design_sha256'] = file_hash(root / design_name)
    response = answer_question(ctl, 'budget', {'budget': brief.get('budget', {}), 'plan': plan_name,
        'reason': 'Offline software fixture only; no scientific or paid calls'},
        {'choice': 'custom', 'budget': brief.get('budget', {})}, 'budget')
    ctl.confirm_budget(response, 'confirm-fixture-budget')
    commit(plan)
    ctl.begin_round()

    source = put(root, 'literature/source.txt', 'Synthetic method source for software testing. Its unit is the case, with repeated seeds averaged within case.')
    verification = put(root, 'literature/verification.txt', 'Local synthetic metadata verified against this generated fixture; not a real publication.')
    catalog = {'question_formed_at': '2026-11-03T00:00:00Z',
               'searches': [{'query': 'software fixture AND case means', 'method': 'keyword', 'result_ids': ['L1', 'L2']}],
               'naive_baseline_ids': ['L1'], 'hard_negative_ids': ['L2'],
               'candidates': [{'id': 'L1', 'title': 'Synthetic method record (software fixture)', 'url': 'https://example.invalid/synthetic-L1',
                               'published_at': '2026-10-01T00:00:00Z', 'exposure': 'prospective', 'purpose': 'A naive comparison for software fixture checking',
                               'decision': 'include', 'read_status': 'read', 'read_path': source, 'verification': {'status': 'primary_checked', 'evidence': verification},
                               'evidence_spans': [{'path': source, 'quote': 'Its unit is the case, with repeated seeds averaged within case.'}]},
                              {'id': 'L2', 'title': 'Synthetic hard negative (software fixture)', 'url': 'https://example.invalid/synthetic-L2',
                               'published_at': '2026-10-01T00:00:00Z', 'exposure': 'prospective', 'purpose': 'Identify a topically related incompatible unit',
                               'decision': 'exclude', 'exclusion_reason': 'Treats repeated seeds as independent cases', 'read_status': 'registered'}]}
    catalog_name = put(root, 'literature/catalog.json', catalog)
    bibliography = put(root, 'manuscript/references.bib', '@misc{fixture, title={Synthetic method record, software fixture only}, author={Offline Fixture}, year={2026}, journal={Software fixture}}\n')
    identity = {'title': 'Synthetic method record, software fixture only', 'authors': ['Offline Fixture'], 'year': 2026, 'venue': 'Software fixture', 'identifier': 'fixture:L1'}
    put(root, verification, 'Primary metadata for the SOFTWARE FIXTURE only: ' + '; '.join([identity['title'], 'Offline Fixture', '2026', identity['venue'], identity['identifier']]))
    catalog['candidates'][0]['title'] = identity['title']
    put(root, catalog_name, catalog)
    span = {'path': source, 'quote': 'Its unit is the case, with repeated seeds averaged within case.'}
    citation = {'key': 'fixture', 'candidate_id': 'L1', **identity, 'url': 'https://example.invalid/synthetic-L1', 'taxonomy_cell': 'case-comparison',
                'depth': 'B', 'status': 'verified', 'claims': [{'id': 'LC1', 'text': span['quote'], 'evidence_spans': [span]}]}
    citation_plan = put(root, 'literature/citation-plan.jsonl', json.dumps(citation) + '\n')
    citation_verification = put(root, 'literature/citation-verification.json', {'records': [{'key': 'fixture', 'status': 'verified', 'checked_at': '2026-11-03T02:00:00Z',
          'identity': identity, 'source_sha256': file_hash(root / source), 'evidence': [verification], 'claim_checks': [{'claim_id': 'LC1', 'status': 'supported', 'evidence_spans': [span]}]}], 'batches': []})
    commit(packet('literature', [catalog_name, source, verification, bibliography, citation_plan, citation_verification], catalog=catalog_name,
                  bibliography=bibliography, citation_plan=citation_plan, citation_verification=citation_verification))

    if survey:
        return run_survey_tail(root, ctl, commit, contexts, plan_items, source, span, template, bibliography, citation_plan, citation_verification, brief)

    observed = json.loads(subprocess.check_output([sys.executable, str(root / evaluator)], text=True))
    runs = []
    for record in observed:
        key = record['case_id'] + '-' + record['condition'] + '-' + str(record['seed'])
        row = dict(id=key, **record, raw='experiment/raw/' + key + '.json')
        if row['status'] != 'completed':
            row['reason'] = 'Explicit simulated software-fixture ' + row['status']
            row['failure_kind'] = 'measurement'
        put(root, row['raw'], {k: v for k, v in row.items() if k not in {'id', 'raw'}})
        runs.append(row)
    results = {'design': design_name, 'design_sha256': file_hash(root / design_name), 'started_at': '2026-11-03T01:00:00Z',
               'outcome': 'negative', 'reported_denominator': len(runs), 'runs': runs}
    results_name = put(root, 'experiment/results.json', results)
    summary = put(root, 'experiment/summary.md', 'SOFTWARE FIXTURE. Candidate is worse on observed cases. All failures/stops retained:\n' + '\n'.join(r['id'] for r in runs if r['status'] != 'completed'))
    experiment = packet('experiment', [design_name, evaluator, results_name, summary, *[r['raw'] for r in runs]], design=design_name, results=results_name, summary=summary)
    put(root, 'experiment/packet.json', experiment)
    ctl.execute('W-experiment', 'execute-experiment')
    commit(experiment)

    rendered = ctl.execute('W-figures', 'execute-figures')['effects'][0]['effect']
    statistics_name, trace_name = 'figures/summary.tex.stats.json', 'figures/results.pdf.trace.json'
    figures = packet('figures', [design_name, evaluator, results_name, *[r['raw'] for r in runs], statistics_name, 'figures/summary.tex', 'figures/results.pdf', trace_name],
                     design=design_name, results=results_name, statistics=statistics_name, table='figures/summary.tex', figure='figures/results.pdf', trace=trace_name, caption=rendered['caption'])
    commit(figures)

    stats = json.loads((root / statistics_name).read_text())
    mean = stats['conditions']['candidate']['mean']
    sentence = f'The candidate mean error was {mean:.6f} points.'
    limitations = [{'kind': 'scope', 'text': 'These software measurements describe only the planned fixture cases.'},
                   {'kind': 'small_sample', 'text': 'Four independent cases do not establish a population effect.'},
                   {'kind': 'failure', 'text': 'All simulated failed and stopped attempts remain in the denominator with the frozen penalty.'},
                   {'kind': 'adverse', 'text': 'The candidate was worse on these paired fixture cases; the negative outcome is retained.'}]
    assertions = {'assertions': [{'id': 'A1', 'claim_id': 'C1', 'text': sentence, 'statistics': statistics_name, 'measure': 'candidate.mean', 'value': mean,
                                 'unit': 'points', 'denominator': 4, 'condition': 'all-planned-cases', 'strength': 'descriptive', 'kind': 'numeric'}],
                  'limitations': limitations, 'prose_review': {'problem': 'Compare fixture transformations', 'choice': 'Fixed full-denominator pairing', 'finding': sentence, 'scope': 'Software fixture only'}, 'revisions': []}
    facts = put(root, 'manuscript/assertions.json', assertions)

    manuscript = r'''\documentclass{article}
\usepackage{graphicx}
\usepackage[labelsep=period]{caption}
\title{Offline workflow fixture}
\begin{document}\maketitle
\begin{abstract}This is a software fixture, not a research paper or evidence of scientific readiness.\end{abstract}
\section{Question and method}We compare two fixed transformations using a frozen local evaluator. Repeated seeds are averaged within each of four cases. The synthetic source is a software record~\cite{fixture}.
\section{Results}
''' + sentence + '\n' + '\n'.join(x['text'] for x in limitations) + r'''
Figure 1 and Table 1 report the same full-denominator observations.
\begin{figure}[ht]\centering\includegraphics[width=.65\linewidth]{../figures/results.pdf}\caption{Paired software fixture errors with case-level standard deviations.}\label{fig:F1}\end{figure}
\begin{table}[ht]\centering\input{../figures/summary.tex}\caption{Full-denominator fixture statistics.}\label{tab:T1}\end{table}
\begin{thebibliography}{1}\bibitem{fixture} Offline Fixture. Synthetic method record. Software testing only, 2026.\end{thebibliography}
\end{document}
'''
    tex = put(root, 'manuscript/main.tex', manuscript)
    ctl.execute('W-writing', 'execute-writing')
    write = packet('write', [tex, 'manuscript/main.pdf', facts, bibliography, statistics_name, template], manuscript=tex, manuscript_tex=tex,
                   manuscript_pdf='manuscript/main.pdf', assertions=facts, bibliography=bibliography, template_source=template, template_sha256=file_hash(root / template), section_packets=[{'id': 'S1', 'path': tex, 'purpose': 'Report fixture evidence and limits', 'claim_ids': ['C1'], 'citation_keys': ['fixture'], 'evidence': [statistics_name], 'baseline': None, 'line_limit_exception': None}])
    commit(write)

    return finish(root, ctl, commit, contexts, plan_items, tex, [results_name, statistics_name], bibliography, citation_plan, citation_verification)


def finish(root, ctl, commit, contexts, plan_items, tex, evidence, bibliography, citation_plan, citation_verification, *, survey=False):
    plan_name, snapshot = 'plan/figure_table_plan.json', 'plan/snapshot.json'
    binding_inputs = []
    for key, work, asset, location in [('F1', 'W-figures', 'figures/results.pdf', r'\label{fig:F1}'),
                                       ('T1', 'W-figures', 'figures/summary.tex', r'\label{tab:T1}'),
                                       ('S1', 'W-writing', tex, r'\section{Results}'),
                                       ('M1', 'W-writing', 'manuscript/main.pdf', r'\title{Offline workflow fixture}')]:
        binding_inputs.append({'requirement_id': key, 'work_id': work, 'assets': [asset], 'evidence': evidence,
                               'manuscript': tex, 'location': location})
    ctl.bind_all(binding_inputs, 'bind-fixture-items')
    bindings = ctl.load()['bindings']
    review_report = put(root, 'review/report.md', 'SOFTWARE FIXTURE reviewer packet. These are scripted checks, not an independent model review. All planned items and in-scope source or numerical evidence are preserved.')
    review = packet('review', [review_report, plan_name, snapshot, tex, 'manuscript/main.pdf'], reviewer='fixture-reviewer',
                    figure_table_plan=plan_name, plan_snapshot=snapshot, manuscript_pdf='manuscript/main.pdf',
                    r6_reviews=[{'id': i['id'], 'verdict': 'fulfilled', 'reason': 'Present in compiled software fixture with raw evidence', 'evidence_locations': [r'Figure 1' if i['kind'] == 'figure' else 'Table 1']} for i in plan_items],
                    gates={key: {'verdict': 'pass', 'evidence': [review_report], 'reason': 'Scripted ' + key + ' checks on synthetic evidence only; not scientific acceptance'} for key in ['G1', 'G2', 'G3', 'G4', 'G5', 'R6']},
                    requirement_checks=[{'requirement_id': key, 'binding_sha256': val['binding_sha256'], 'evidence_hashes': val['hashes'],
                                         'verdict': 'fulfilled', 'rationale': 'Scripted software fixture correspondence check; no independent scientific review claim'} for key, val in bindings.items()])
    if survey:
        review['gates']['G2'].update(verdict='not_applicable', reason=ctl.load()['brief']['not_applicable']['experiment'])
    revision = put(root, 'review/revision.json', {'status': 'initial_draft', 'changes': []})
    prepared = ctl.prepare_review('review/request.json')
    judgment = {key: review[key] for key in ['reviewer', 'gates', 'r6_reviews', 'requirement_checks']}
    judgment.update(request_sha256=prepared['request_sha256'], reports=[review_report], findings=[],
                    reviewer_identity={'kind': 'software_fixture', 'independent_model_review': 'not performed'})
    put(root, 'review/judgment.json', judgment)
    ctl.assemble_review({'request': 'review/request.json', 'judgment': 'review/judgment.json', 'target': 'review/assembled.json',
                         'completion': {'revision': [revision], 'snapshot': 'review/version-snapshot.json', 'r6_report': 'review/r6-mechanical.json'}})
    review = json.loads((root/'review/assembled.json').read_text())
    version_snapshot = review['completion']['snapshot'][0]
    commit(review)
    delivery_context = ctl.context()
    assert delivery_context['stage'] == 'delivery' and delivery_context['knowledge']
    contexts.append({'stage': 'delivery', 'knowledge_sha256': delivery_context['knowledge_sha256']})
    request = ctl.prepare_round([review_report, version_snapshot])
    exported = ctl.deliver({'request_sha256': request['request_sha256'], 'progress': 'verified_progress', 'reason': 'Complete offline software evidence was validated; not a scientific result.', 'status': 'active', 'phase': 'integration'}, 'export-fixture')
    ctl.verify_export(exported['manifest'])
    put(root, 'fixture-report.json', {'status': exported['status'], 'contexts': contexts,
                                     'kind': 'source_synthesis' if survey else 'empirical',
                                     'negative_results_retained': 'not_applicable' if survey else True, 'model_calls': 0, 'remote_jobs': 0,
                                     'export': exported, 'independent_model_review': 'not performed'})
    return exported


def run_survey_tail(root, ctl, commit, contexts, plan_items, source, span, template, bibliography, citation_plan, citation_verification, brief):
    reason = brief['not_applicable']['experiment']
    note = put(root, 'experiment/not-applicable.md', reason)
    commit(packet('experiment', [note], not_applicable=reason))
    diagram = put(root, 'figures/synthesis.tex', r'''\documentclass{article}
\begin{document}\pagestyle{empty}
\fbox{Synthetic source: repeated seeds $\rightarrow$ case mean $\rightarrow$ case unit}
\end{document}
''')
    classification = put(root, 'literature/classification.tex', r'\begin{tabular}{ll}Source&Statistical unit\\Synthetic record&Case (seeds averaged)\\\end{tabular}' + '\n')
    ctl.execute('W-figures', 'execute-figures')
    trace = put(root, 'figures/evidence-trace.json', {'kind': 'evidence_diagram', 'figure_sha256': file_hash(root / 'figures/results.pdf'),
           'table_sha256': file_hash(root / 'figures/summary.tex'), 'source_hashes': {source: file_hash(root / source)}, 'claim_ids': ['LC1'], 'citation_keys': ['fixture']})
    commit(packet('figures', [diagram, classification, source, 'figures/results.pdf', 'figures/summary.tex', trace], mode='evidence_diagram',
                  figure='figures/results.pdf', table='figures/summary.tex', trace=trace, caption='Source-bounded classification of a synthetic record, with no comparative result.',
                  claim_ids=['LC1'], citation_keys=['fixture'], evidence_spans=[span]))
    sentence = span['quote']; scope = 'This synthesis covers one synthetic record; it establishes no general scientific result.'
    facts = put(root, 'manuscript/assertions.json', {'assertions': [{'id': 'A1', 'claim_id': 'LC1', 'kind': 'source_claim', 'strength': 'source_bounded',
               'text': sentence, 'citation_keys': ['fixture']}], 'limitations': [{'kind': 'scope', 'text': scope}],
               'prose_review': {'problem': 'Classify source evidence', 'choice': 'Preserve the verified unit', 'finding': sentence, 'scope': scope}, 'revisions': []})
    tex = put(root, 'manuscript/main.tex', r'''\documentclass{article}
\usepackage{graphicx}
\usepackage[labelsep=period]{caption}
\title{Offline workflow fixture}
\begin{document}\maketitle
\begin{abstract}A software test of non-empirical synthesis, not a scientific survey.\end{abstract}
\section{Question and method}We classify the evidence unit in one verified synthetic source~\cite{fixture}. No new empirical evaluation is in scope.
\section{Results}
''' + sentence + '\n' + scope + r'''
Figure 1 and Table 1 describe the bounded classification.
\begin{figure}[ht]\centering\includegraphics[width=.8\linewidth]{../figures/results.pdf}\caption{Verified source evidence unit.}\label{fig:F1}\end{figure}
\begin{table}[ht]\centering\input{../figures/summary.tex}\caption{Source classification, without measured results.}\label{tab:T1}\end{table}
\begin{thebibliography}{1}\bibitem{fixture} Offline Fixture. Synthetic method record. Software testing only, 2026.\end{thebibliography}
\end{document}
''')
    ctl.execute('W-writing', 'execute-writing')
    commit(packet('write', [tex, 'manuscript/main.pdf', facts, bibliography, source, template], manuscript=tex, manuscript_tex=tex,
                  manuscript_pdf='manuscript/main.pdf', assertions=facts, bibliography=bibliography, template_source=template, template_sha256=file_hash(root / template),
                  section_packets=[{'id': 'S1', 'path': tex, 'purpose': 'Report a bounded verified source classification', 'claim_ids': ['LC1'], 'citation_keys': ['fixture'],
                                    'evidence': [source], 'baseline': None, 'line_limit_exception': None}]))
    return finish(root, ctl, commit, contexts, plan_items, tex, [source, citation_verification], bibliography, citation_plan, citation_verification, survey=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', required=True); parser.add_argument('--undecided', action='store_true'); parser.add_argument('--survey', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.run, undecided=args.undecided, survey=args.survey), indent=2))
