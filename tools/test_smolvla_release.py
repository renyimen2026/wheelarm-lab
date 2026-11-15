import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from validate_smolvla_release import validate,digest


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.root=Path(self.temporary.name)
        self.checkpoint=self.root/'model';self.checkpoint.mkdir()
        (self.checkpoint/'model.safetensors').write_bytes(b'test-only')
        self.executor=self.root/'executor.py';self.executor.write_text('test-only')
        self.trials=[]
        for i in range(5):
            p=self.root/f'trial_{i}';(p/'policy').mkdir(parents=True)
            plan={'checkpoint':str(self.checkpoint),'dataset':str(self.root/'dataset'),
                'initial_arm':[-1.05+i*.001,2.1,-1.05],'ign_partition':f'trial_{i}',
                'action_steps':10}
            plan['runtime_source_sha256']={str(self.executor):digest(self.executor)}
            before={'status':'detected','top_base_m':.7,'center_base_m':[.66,0,.65],'radius_m':.025}
            after=dict(before,top_base_m=.78)
            samples=[{'elapsed_seconds':t,'joints_stamp_ns':int((t+1)*1e9),
                      'measurement':after,'state':[0,-1,2,-1,.05]} for t in (0,5,10)]
            report={'checkpoint':str(self.checkpoint),'checkpoint_sha256':digest(self.checkpoint/'model.safetensors'),
                'status':'completed','error':None,'end_reason':'lift_and_hold_verified','task_success':True,
                'sampling_seed':i,'ensemble_size':4,'cup_before':before,'cup_after':after,'hold_samples':samples}
            for name,value in [('trial_plan.json',plan),('trial_result.json',{'success':True}),
                               ('policy/live_report.json',report)]:
                (p/name).write_text(json.dumps(value))
            self.trials.append(p)

    def tearDown(self):self.temporary.cleanup()

    def test_five_fresh_trials_make_a_pinned_release(self):
        release=validate(self.trials,self.root)
        self.assertEqual(release['status'],'ready_for_user_simulation_acceptance')
        self.assertEqual(len(release['verified_trials']),5)
        self.assertEqual(release['ensemble_size'],4)

    def test_duplicates_or_missing_trials_do_not_pass(self):
        for values in [self.trials[:4],self.trials[:4]+self.trials[:1]]:
            with self.assertRaises(ValueError):validate(values,self.root)

    def test_stale_success_flag_cannot_hide_a_dropped_cup(self):
        p=self.trials[0]/'policy/live_report.json'
        r=json.loads(p.read_text());r['hold_samples'][-1]['measurement']['top_base_m']=.7
        p.write_text(json.dumps(r))
        with self.assertRaises(ValueError):validate(self.trials,self.root)

    def test_changed_weights_are_rejected(self):
        (self.checkpoint/'model.safetensors').write_bytes(b'changed')
        with self.assertRaises(ValueError):validate(self.trials,self.root)

    def test_changed_execution_source_is_rejected(self):
        self.executor.write_text('new implementation')
        with self.assertRaises(ValueError):validate(self.trials,self.root)

    def test_mixed_action_time_scales_are_rejected(self):
        p=self.trials[0]/'trial_plan.json';q=self.trials[0]/'policy/live_report.json'
        plan=json.loads(p.read_text());plan['action_stride']=5;p.write_text(json.dumps(plan))
        with self.assertRaises(ValueError):validate(self.trials,self.root)
        report=json.loads(q.read_text());report['action_stride']=5;q.write_text(json.dumps(report))
        with self.assertRaises(ValueError):validate(self.trials,self.root)

    def test_gripper_reference_mode_must_match_evidence_and_all_cases(self):
        p=self.trials[0]/'trial_plan.json';q=self.trials[0]/'policy/live_report.json'
        plan=json.loads(p.read_text());plan['gripper_start_source']='reference';p.write_text(json.dumps(plan))
        with self.assertRaises(ValueError):validate(self.trials,self.root)
        report=json.loads(q.read_text());report['gripper_start_source']='reference';q.write_text(json.dumps(report))
        with self.assertRaises(ValueError):validate(self.trials,self.root)
        for trial in self.trials:
            for name in ['trial_plan.json','policy/live_report.json']:
                path=trial/name;value=json.loads(path.read_text())
                value['gripper_start_source']='reference';path.write_text(json.dumps(value))
        self.assertEqual(validate(self.trials,self.root)['gripper_start_source'],'reference')

    def test_user_entrypoint_replays_validated_gripper_source(self):
        from run_released_cup_policy import main
        release=validate(self.trials,self.root)
        release['gripper_start_source']='reference'
        manifest=self.root/'release.json';manifest.write_text(json.dumps(release))
        captured=[]
        def run():captured.extend(sys.argv);return 0
        with patch.object(sys,'argv',['run','--release',str(manifest),'--output-dir',str(self.root/'new_trial')]), \
                patch.dict(sys.modules,{'probe_smolvla_live':SimpleNamespace(main=run)}):
            self.assertEqual(main(),0)
        self.assertEqual(captured[captured.index('--gripper-start-source')+1],'reference')
        self.assertEqual(captured[captured.index('--ensemble-size')+1],'4')


if __name__=='__main__':unittest.main(verbosity=2)
