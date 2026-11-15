"""Exercise the real orchestration/report writer with mocked GPU and ROS transport."""
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_smolvla_live as live

torch, forward = live.torch, live.forward
real_generator, real_randn = torch.Generator, torch.randn


class FakeSession:
    def __init__(self, root, behavior):
        self.root = root
        root.mkdir()
        self.sequence, self.motions, self.behavior = 0, 0, behavior
        self.closed = False
        self.captures = 0
        self.motion_requests = []

    def request(self, value):
        number = self.sequence
        self.sequence += 1
        op = value['operation']
        if op == 'measure_cup':
            if self.behavior == 'occluded':return {'measurement':{'status':'unavailable'}}
            height = .78 if self.behavior=='lift_detected' and self.motions else .7
            return {'measurement': {'status':'detected', 'top_base_m':height, 'radius_m':.025, 'center_base_m':[.66,0.,.65]}}
        if op == 'capture':
            self.captures += 1
            directory = Path(value['output_dir'])/'frame'
            directory.mkdir(parents=True)
            Image.fromarray(np.zeros((480,640,3), dtype=np.uint8)).save(directory/'rgb.png')
            meta = {'kind':'observation_only', 'state':{'names':list(forward.STATE_NAMES), 'units':list(forward.STATE_UNITS), 'values':[0.,-.6,.9,-.3,.06]},
                    'image':{'file':'rgb.png', 'stamp_ns':1000000000}, 'joints_stamp_ns':1000000000,
                    'image_age_ms':20., 'joints_age_ms':20., 'timestamp_difference_ms':0.}
            if self.behavior=='lift_detected' and self.motions:
                meta['state']['values'][4]=.05
            meta['joints_stamp_ns']=meta['image']['stamp_ns']=self.captures*5_000_000_000
            (directory/'observation.json').write_text(json.dumps(meta))
            return {'observation_directory':str(directory)}
        self.motion_requests.append(value)
        self.motions += 1
        result = {'status':'completed', 'execution_attempted':True, 'actual_after':[0.,-.6,.9,-.3,.06], 'gripper_tcp_base_m':[.4,0.,.6]}
        if self.behavior == 'deadline':
            result = {'status':'completed', 'execution_attempted':False, 'motion_skipped':True}
        if self.behavior == 'fail_second' and self.motions == 2:
            result.update(status='failed', error='injected execution failure')
        (self.root/f'result_{number:03d}.json').write_text(json.dumps(result))
        if result['status']=='failed':
            raise RuntimeError(result['error'])
        return result

    def close(self):
        self.closed = True


class LoopTests(unittest.TestCase):
    def run_case(self, behavior, hold=False, horizon=50,ensemble=1,stride=1,gripper_source='actual'):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root/'output'
            config = SimpleNamespace(device='cpu',chunk_size=horizon)
            data = SimpleNamespace(manifest={'instruction':'拿起桌上的杯子并保持夹持'})
            policy = Mock()
            policy.to.return_value = policy
            policy.eval.return_value = policy
            policy.predict_action_chunk.return_value = torch.arange(ensemble).float()[:,None,None].expand(ensemble,horizon,5)
            if stride==5:
                policy.predict_action_chunk.return_value=policy.predict_action_chunk.return_value.clone()
                policy.predict_action_chunk.return_value[:,:,0]+=torch.arange(horizon)*.001
            sessions = []
            def session_factory(path):
                session = FakeSession(path, behavior)
                sessions.append(session)
                return session
            args = ['probe',str(root/'dataset'),'--checkpoint',str(root/'checkpoint'), '--assets-dir',str(root/'assets'),
                    '--output-dir',str(output),'--execute','--trial-seconds','90','--max-cycles','2','--action-steps','10']
            if behavior in ('lift_detected','occluded'):args.extend(['--verify-every','1'])
            if hold:args.extend(['--hold-seconds','10'])
            args.extend(['--ensemble-size',str(ensemble)])
            args.extend(['--action-stride',str(stride)])
            args.extend(['--gripper-start-source',gripper_source])
            clock=[0.]
            def now():
                clock[0]+=.001
                return clock[0]
            def sleep(seconds):clock[0]+=seconds
            with patch.object(sys,'argv',args), patch.object(live.validation,'prepare_inputs',return_value=(root/'checkpoint',config,data,{},{})), \
                 patch.object(live.validation,'checkpoint_digest',return_value='test-hash'), \
                 patch.object(forward,'load_policy',return_value=policy), \
                 patch.object(forward,'make_smolvla_pre_post_processors',return_value=(lambda b:b,lambda p:p)), \
                 patch.object(live,'RosProbeSession',side_effect=session_factory), \
                 patch.object(torch.cuda,'is_available',return_value=True), patch.object(torch.cuda,'is_bf16_supported',return_value=True), \
                 patch.object(torch.cuda,'synchronize'), patch.object(torch,'autocast',side_effect=lambda *a,**k:nullcontext()), \
                 patch.object(torch,'Generator',side_effect=lambda **k:real_generator(device='cpu')), \
                 patch.object(torch,'randn',side_effect=lambda *a,**k:real_randn(*a,**dict(k,device='cpu'))), \
                 (patch.object(live.time,'monotonic',side_effect=now) if hold else nullcontext()), \
                 (patch.object(live.time,'sleep',side_effect=sleep) if hold else nullcontext()):
                code = live.main()
            report = json.loads((output/'live_report.json').read_text())
            self.assertTrue(sessions[0].closed)
            self.assertTrue(all(len(q['predictions_physical']) == 10 for q in sessions[0].motion_requests))
            self.assertEqual(report['action_steps_per_cycle'], 10)
            for request in sessions[0].motion_requests:
                self.assertEqual(request['action_step_seconds'],stride/10)
                self.assertEqual(request['gripper_start_source'],gripper_source)
                if stride==5:
                    self.assertAlmostEqual(request['first_prediction_physical'][0],.004)
                    self.assertAlmostEqual(request['predictions_physical'][-1][0],.049)
            self.assertEqual(len(list(output.glob('prediction_*.json'))), 1 if behavior in ('deadline','lift_detected') else 2)
            return code, report

    def test_two_cycles_finish_without_recreating_prediction_files(self):
        code, report = self.run_case('success')
        self.assertEqual(code,0)
        self.assertEqual(report['completed_cycles'],2)
        self.assertEqual(report['end_reason'],'cycle_limit')
        self.assertFalse(report['lift_evidence']['lift_verified'])

    def test_reference_start_is_forwarded_and_recorded_without_changing_policy_inputs(self):
        code,report=self.run_case('success',gripper_source='reference')
        self.assertEqual(code,0)
        self.assertEqual(report['gripper_start_source'],'reference')
        np.testing.assert_allclose(report['samples'][0]['state_physical'],[0.,-.6,.9,-.3,.06],atol=1e-7)

    def test_stride_five_uses_future_half_second_points_without_compressing_time(self):
        code,report=self.run_case('success',stride=5)
        self.assertEqual(code,0)
        self.assertEqual(report['action_stride'],5)
        self.assertEqual(report['samples'][0]['executed_prediction_indices'],list(range(4,50,5)))
        self.assertEqual(report['samples'][0]['executed_physical_offsets_seconds'],[.5*(i+1) for i in range(10)])

    def test_child_deadline_is_a_clean_stop_without_motion(self):
        code, report = self.run_case('deadline')
        self.assertEqual(code,0)
        self.assertEqual(report['completed_cycles'],0)
        self.assertFalse(report['robot_commands_sent'])
        self.assertEqual(report['end_reason'],'time_limit_before_dispatch')

    def test_execution_failure_preserves_completed_cycle_and_failed_result(self):
        code, report = self.run_case('fail_second')
        self.assertEqual(code,1)
        self.assertEqual(report['status'],'stopped')
        self.assertEqual(report['completed_cycles'],1)
        self.assertTrue(report['robot_commands_sent'])
        self.assertEqual(report['samples'][1]['motion']['status'],'failed')

    def test_verified_lift_stops_new_actions_and_is_rechecked_at_end(self):
        code, report = self.run_case('lift_detected')
        self.assertEqual(code,0)
        self.assertEqual(report['completed_cycles'],1)
        self.assertEqual(report['end_reason'],'lift_geometry_verified')
        self.assertTrue(report['lift_evidence']['lift_verified'])
        self.assertIsNone(report['lift_evidence']['task_success'])

    def test_success_requires_lift_plus_ten_second_hold_without_further_actions(self):
        code,report=self.run_case('lift_detected',hold=True)
        self.assertEqual(code,0)
        self.assertTrue(report['task_success'])
        self.assertEqual(report['end_reason'],'lift_and_hold_verified')
        self.assertEqual(report['completed_cycles'],1)
        self.assertEqual(len(report['hold_samples']),3)
        self.assertGreaterEqual(report['hold_evidence']['duration_sim_seconds'],10)

    def test_ten_step_policy_uses_matching_noise_and_physical_horizon(self):
        code,report=self.run_case('success',horizon=10)
        self.assertEqual(code,0)
        self.assertEqual(report['policy_chunk_size'],10)
        self.assertEqual(len(report['samples'][0]['prediction_physical']),10)
        self.assertEqual(report['samples'][0]['physical_action_offsets_seconds'][-1],1.)

    def test_four_member_inference_preserves_members_and_uses_their_mean(self):
        code,report=self.run_case('success',horizon=10,ensemble=4)
        self.assertEqual(code,0)
        record=report['samples'][0]
        self.assertEqual(report['ensemble_size'],4)
        self.assertEqual(len(record['ensemble_predictions_physical']),4)
        self.assertTrue(np.allclose(record['prediction_physical'],1.5))

    def test_missing_periodic_measurement_does_not_claim_success(self):
        code, report = self.run_case('occluded')
        self.assertEqual(code,0)
        self.assertEqual(report['completed_cycles'],2)
        self.assertEqual(report['end_reason'],'cycle_limit')
        self.assertIsNone(report['lift_evidence']['lift_verified'])


if __name__=='__main__':
    unittest.main(verbosity=2)
