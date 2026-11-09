"""Reusable production behavior with variable inputs, not the kNN recipe."""
import copy
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import production, science
from lib.experiments import ExperimentRecorder, MeasurementFailure
from lib.runtime import Controller, Refusal, read_json, file_hash
from run_fixture import make_brief, put

CLI = Path(__file__).resolve().parents[1]/'scripts/research.py'


def design():
    return {'experiment_id':'latency','claim_id':'latency-claim','hypothesis':'Compare a bounded transformation',
        'independent_variable':'transformation','dependent_variable':'latency','control_variables':['same inputs'],
        'metric':{'name':'latency','unit':'ms','direction':'lower'},'statistical_unit':'case',
        'cases':['site-a','site-b','site-c'],'seeds':[7],'conditions':['baseline','candidate'],
        'failure_policy':'penalty','failure_value':100,'statistics':{'estimator':'mean','interval':'sd','paired':True},
        'data_roles':[{'id':'heldout','role':'test','used_for_selection':False}], 'planned_at':'2025-01-01T00:00:00Z'}


class ProductionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'study';self.root.mkdir()

    def cli(self,*args,cwd=None):
        result=subprocess.run([sys.executable,str(CLI),'--run',str(self.root),*args],cwd=cwd,capture_output=True,text=True)
        return result,json.loads(result.stdout)

    def test_cli_owns_request_persistence_and_paths_from_another_cwd(self):
        put(self.root,'brief.json',make_brief())
        elsewhere=Path(self.temp.name)/'elsewhere';elsewhere.mkdir()
        result,value=self.cli('init','--brief','brief.json',cwd=elsewhere)
        self.assertEqual(0,result.returncode,value)
        self.cli('approve','--brief-sha256',value['brief_sha256'],'--actor','software fixture',cwd=elsewhere)
        self.cli('begin-round',cwd=elsewhere)
        result,request=self.cli('prepare-round',cwd=elsewhere)
        stored=self.root/'control/round-decisions'/(request['request_sha256']+'-request.json')
        self.assertEqual(request,read_json(stored))
        put(self.root,'control/round-decisions/decision.json',{'request_sha256':request['request_sha256'],
            'progress':'no_progress','reason':'Only a software control check','status':'active'})
        result,value=self.cli('apply-round','--decision','control/round-decisions/decision.json','--operation-id','close',cwd=elsewhere)
        self.assertEqual(0,result.returncode,value)
        self.assertTrue(list((self.root.parent/'study-cli').glob('*.json')))
        self.assertFalse((self.root/'control/cli').exists())
        put(elsewhere,'brief.json',{'different':'input'})
        result,value=self.cli('init','--brief','brief.json',cwd=elsewhere)
        self.assertEqual(2,result.returncode)
        self.assertIn('Ambiguous',value['reason'])

    def test_tex_variants_preserve_comments_text_and_explicit_caption_order(self):
        put(self.root,'tables/data.tex','Visible table text.\n')
        source=r'''\documentclass{article}
\usepackage{graphicx}
\usepackage{hyperref}
\begin{document}
The argument refers to Fig~\ref{plot:flow}.

\section{Methods}
\input{../tables/data.tex} % \input{missing-comment-file}
\begin{figure}[ht]
\includegraphics{image.pdf}
\caption{Actual task caption}\label{plot:flow}
\end{figure}
\section{Discussion}
\end{document}
'''
        put(self.root,'manuscript/body.tex',source)
        packet={'source':'manuscript/body.tex','target':'manuscript/no-graphics.tex','options':
            {'graphics':'omit','inline_inputs':True,'link_borders':'hidden','captions_after_first_reference':True,
             'flush_before':[r'\section{Discussion}']}}
        value=production.run('tex-prepare',packet,self.root)
        result=(self.root/packet['target']).read_text()
        self.assertIn('Visible table text.',result)
        self.assertIn(r'% \input{missing-comment-file}',result)
        self.assertNotIn(r'\includegraphics',result)
        self.assertLess(result.index(r'\caption'),result.index(r'\section{Methods}'))
        self.assertEqual({'manuscript/body.tex','tables/data.tex'},set(value['input_hashes']))
        self.assertIn('hidelinks',result)
        self.assertIn('\\clearpage\n\\section{Discussion}',result)
        production.run('tex-prepare',packet,self.root)
        changed=copy.deepcopy(packet);changed['options']['graphics']='keep'
        changed['options']['captions_after_first_reference']=False
        with self.assertRaisesRegex(Refusal,'versioned'):
            production.run('tex-prepare',changed,self.root)

    def test_raster_exports_use_input_policy_and_keep_transparency_honest(self):
        from PIL import Image
        Image.new('RGBA',(123,77),(20,30,40,255)).save(self.root/'input.png',dpi=(144,144))
        value=production.run('image-export',{'source':'input.png','target':'out/figure.tif',
            'options':{'mode':'RGB','dpi':144,'min_dpi':140,'width_pixels':123,'height_pixels':77}},self.root)
        self.assertEqual([123,77],value['observed']['pixels']);self.assertEqual('RGB',value['observed']['mode'])
        self.assertTrue(all(type(x) is float for x in value['observed']['dpi']))
        Image.new('RGBA',(17,13),(20,30,40,100)).save(self.root/'transparent.png')
        with self.assertRaisesRegex(Refusal,'Nonopaque'):
            production.run('image-export',{'source':'transparent.png','target':'out/other.tif','options':{'mode':'RGB','dpi':600}},self.root)
        self.assertFalse((self.root/'out/other.tif').exists())
        with self.assertRaisesRegex(Refusal,'controller-owned'):
            production.run('image-export',{'source':'input.png','target':'.paper/bad.tif','options':{}},self.root)

    def test_bibliography_copies_compound_name_components_and_only_public_fields(self):
        put(self.root,'publisher.xml','<article><article-meta><contrib-group><contrib contrib-type="author"><name><surname>Van Example</surname><given-names>Ada</given-names></name></contrib></contrib-group></article-meta></article>')
        value=production.run('bibliography',{'target':'refs.bib','entries':[{'key':'source-x','jats':'publisher.xml',
            'title':'A source','year':2011,'journal':'Methods','volume':'5','pages':'1--9','metadata':{'year_basis':'year precision only'}}]},self.root)
        text=(self.root/'refs.bib').read_text()
        self.assertIn('Van Example, Ada',text);self.assertNotIn('year_basis',text)
        self.assertNotIn('note =',text)
        self.assertEqual(['Van Example, Ada'],value['identities'][0]['authors'])
        before,after=science.publication_bounds('2011')
        self.assertEqual(2011,before.year);self.assertEqual(2011,after.year)

    def test_missing_dependencies_are_not_available(self):
        value=production.run('environment',{'dependencies':['autoresearch-nonexistent-unit-test-package']},self.root)
        self.assertEqual('missing_dependencies',value['status'])
        self.assertEqual(1,len(value['missing']))

    def test_cli_keeps_third_party_diagnostics_out_of_json_stdout(self):
        put(self.root,'probe.json',{'dependencies':[]})
        code='''import json,runpy,sys
sys.path.insert(0,sys.argv[1])
from lib import production
def noisy(packet,root):
 print('third-party diagnostic')
 return {'status':'available'}
production.TOOLS['environment']=noisy
sys.argv=[sys.argv[2],'--run',sys.argv[3],'artifact','--name','environment','--packet','probe.json']
runpy.run_path(sys.argv[0],run_name='__main__')
'''
        result=subprocess.run([sys.executable,'-c',code,str(CLI.parents[1]),str(CLI),str(self.root)],capture_output=True,text=True)
        self.assertEqual(0,result.returncode,result.stderr)
        self.assertEqual('available',json.loads(result.stdout)['status'])
        self.assertIn('third-party diagnostic',result.stderr)

    def test_acquisition_reuses_verified_bytes_after_a_later_source_failure(self):
        class Response(io.BytesIO):
            url='https://publisher.test/file';headers={'Content-Type':'application/pdf'}
        sources={'sources':[{'url':'https://publisher.test/one','path':'one.pdf','kind':'pdf'},
                            {'url':'https://publisher.test/two','path':'two.pdf','kind':'pdf'}]}
        with patch('urllib.request.urlopen',side_effect=[Response(b'%PDF-test'),Response(b'<html>login</html>')]):
            with self.assertRaisesRegex(Refusal,'non-PDF'):
                production.run('source-fetch',sources,self.root)
        before=file_hash(self.root/'one.pdf');stamp=(self.root/'one.pdf').stat().st_mtime_ns
        with patch('urllib.request.urlopen',return_value=Response(b'%PDF-second')) as opener:
            result=production.run('source-fetch',sources,self.root)
        self.assertEqual(1,opener.call_count);self.assertTrue(result['sources'][0]['reused'])
        self.assertEqual(before,file_hash(self.root/'one.pdf'));self.assertEqual(stamp,(self.root/'one.pdf').stat().st_mtime_ns)


class RecorderTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        put(self.root,'experiment/design.json',design())

    def test_operational_exception_retains_the_entire_non_knn_grid(self):
        with self.assertRaisesRegex(RuntimeError,'instrument failed'):
            with ExperimentRecorder(self.root) as records:
                for index,key in enumerate(records.grid):
                    with records.attempt(*key) as complete:
                        if index==2: raise RuntimeError('instrument failed')
                        complete(5.0)
        value=read_json(self.root/'experiment/results.json')
        self.assertEqual(6,value['reported_denominator']);self.assertEqual(3,value['attempted_count'])
        self.assertEqual(['completed','completed','failed','stopped','stopped','stopped'],[x['status'] for x in value['runs']])
        self.assertTrue(all(x['value'] is None for x in value['runs'][2:]))
        with self.assertRaisesRegex(ValueError,'operation failures'):
            science.render_statistics(self.root,'experiment/results.json','table.tex','plot.svg')

    def test_successful_grid_uses_three_cases_one_seed_without_penalty_claim(self):
        with ExperimentRecorder(self.root) as records:
            for case,seed,condition in records.grid:
                with records.attempt(case,seed,condition) as complete:
                    complete(4 if condition=='baseline' else 3)
        rendered=science.render_statistics(self.root,'experiment/results.json','table.tex','plot.svg')
        stats=rendered['statistics'];caption=rendered['caption']
        self.assertEqual(3,stats['n_cases']);self.assertEqual(-1,stats['delta']['mean'])
        self.assertNotIn('independent',caption);self.assertNotIn('penalty',caption)
        self.assertIn('sample SD (ddof=1)',caption)
        with self.assertRaisesRegex(Refusal,'Preserve'):
            ExperimentRecorder(self.root)

    def test_explicit_protocol_failure_is_distinct_from_an_operation_failure(self):
        with ExperimentRecorder(self.root) as records:
            for index,key in enumerate(records.grid):
                with records.attempt(*key) as complete:
                    if index==1: raise MeasurementFailure('Frozen protocol says this trial has no admissible measurement')
                    complete(4)
        rendered=science.render_statistics(self.root,'experiment/results.json','table.tex','plot.svg')
        self.assertEqual(1,rendered['statistics']['failed_runs'])
        self.assertEqual(6,rendered['statistics']['planned_runs'])
        self.assertIn('frozen penalty',rendered['caption'])

    def test_error_after_the_last_score_still_blocks_scientific_acceptance(self):
        with self.assertRaisesRegex(RuntimeError,'post-check failed'):
            with ExperimentRecorder(self.root) as records:
                for key in records.grid:
                    with records.attempt(*key) as complete: complete(4)
                raise RuntimeError('post-check failed')
        result=read_json(self.root/'experiment/results.json')
        self.assertEqual('failed',result['execution_status'])
        self.assertTrue(all(row['status']=='completed' for row in result['runs']))
        with self.assertRaisesRegex(ValueError,'operation failures'):
            science.render_statistics(self.root,'experiment/results.json','table.tex','plot.svg')


if __name__=='__main__':unittest.main()
