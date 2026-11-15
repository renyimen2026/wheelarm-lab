"""Physical sequencing diagnostics without GPU, ROS or model weights."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'wheel_arm_manipulation'))
from wheel_arm_manipulation.smolvla_coordination import (
    TCPKinematics, coordination_metrics, summarize_coordination, coordination_plan)


class HeightFK:
    def positions(self, values):
        values=np.asarray(values)
        return np.stack([values[...,0]*0, values[...,0]*0, .599+values[...,0]],axis=-1)


class CoordinationTests(unittest.TestCase):
    def fixture(self):
        target=np.zeros((50,5));target[:,4]=.049;target[:10,4]=.08;target[20:,0]=.08
        state=np.array([0.,0.,0.,0.,.08])
        return target,state

    def metric(self, prediction):
        target,state=self.fixture()
        return coordination_metrics(prediction,target,state,HeightFK(),.599,.049)

    def test_correct_order_and_delayed_closure_are_distinguished(self):
        target,_=self.fixture()
        good=self.metric(target)
        self.assertTrue(good['closed_before_lift'])
        self.assertEqual(good['max_rise_while_open_mm'],0.)
        bad=target.copy();bad[:40,4]=.08
        wrong=self.metric(bad)
        self.assertFalse(wrong['closed_before_lift'])
        self.assertEqual(wrong['closure_state_at_lift'],'too_open')
        self.assertAlmostEqual(wrong['predicted_width_at_lift_mm'],80.)
        self.assertAlmostEqual(wrong['max_rise_while_open_mm'],80.)

    def test_no_lift_is_not_a_pass(self):
        target,_=self.fixture();target[:,0]=0
        result=self.metric(target)
        self.assertIsNone(result['closed_before_lift'])
        summary=summarize_coordination([{'coordination':result}])
        self.assertEqual(summary['closed_before_lift_windows'],0)
        self.assertEqual(summary['no_lift_predicted_windows'],1)

    def test_noise_spike_does_not_count_as_lift_and_overclosure_does_not_pass(self):
        target,_=self.fixture();target[2,0]=.03
        self.assertEqual(self.metric(target)['predicted_lift_offset_s'],2.1)
        target[:,4]=.03
        result=self.metric(target)
        self.assertFalse(result['closed_before_lift'])
        self.assertEqual(result['closure_state_at_lift'],'too_closed')
        self.assertEqual(summarize_coordination([{'coordination':result}])['lift_with_overclosed_gripper_windows'],1)

    def test_invalid_numbers_and_shapes_rejected(self):
        target,state=self.fixture()
        with self.assertRaises(ValueError):self.metric(target[:10])
        target[0,0]=np.nan
        with self.assertRaises(ValueError):self.metric(target)
        target,_=self.fixture()
        with self.assertRaises(ValueError):coordination_metrics(target,target,state,HeightFK(),np.nan,.049)

    def test_numpy_fk_known_chain_and_vectorization(self):
        names=['a','b','c','d'];chain=[]
        for name in names:
            origin=np.eye(4);origin[0,3]=1
            chain.append({'name':name,'kind':'revolute','axis':[0,0,1],'origin':origin.tolist()})
        origin=np.eye(4);origin[0,3]=.2
        chain.append({'name':'tip','kind':'fixed','axis':None,'origin':origin.tolist()})
        fk=TCPKinematics({'joint_names':names,'frame':'base_footprint','tip':'gripper_tcp','chain':chain})
        self.assertTrue(np.allclose(fk.positions([[0,0,0,0,0],[np.pi/2,0,0,0,0]]),[[4.2,0,0],[1,3.2,0]],atol=1e-12))

    def test_transition_selection_and_event_hash_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.mkdir()
            events=[]
            for phase,start,end in [('close',0,2_000_000_000),('lift',2_000_000_000,7_000_000_000)]:
                for event,stamp in [('phase_started',start),('phase_finished',end)]:
                    events.append({'task_id':'t','phase':phase,'event':event,'stamp_ns':stamp})
            path=source/'task_events.jsonl';path.write_text(''.join(json.dumps(v)+'\n' for v in events))
            manifest={'sources':[{'split':'train','task_id':'t','source_episode':str(source),
                'source_file_sha256':{'task_events.jsonl':hashlib.sha256(path.read_bytes()).hexdigest()}}]}
            (root/'source_manifest.json').write_text(json.dumps(manifest))
            rows=[]
            for i in range(70):
                t=i/10
                state=[0. if t<2 else .08,0,0,0,.08 if t<.8 else .049]
                action=[0. if t+.1<2 else .08,0,0,0,.08 if t+.1<.8 else .049]
                rows.append({'source_task_id':'t','source_grid_row':{'stamp_ns':i*100_000_000,
                    'action_stamp_ns':(i+1)*100_000_000,'state':state,'action':action}})
            data=SimpleNamespace(split='train',groups={'t':list(range(21))},mapping=rows)
            plan=coordination_plan(root,data,HeightFK())
            self.assertEqual(len(plan['t']['indices']),2)
            self.assertTrue(set(plan['t']['indices'])<=set(data.groups['t']))
            path.write_text(path.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'changed'):coordination_plan(root,data,HeightFK())


if __name__=='__main__':
    unittest.main(verbosity=2)
