"""No ROS graph or robot commands: persistent-session protocol regression tests."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ros_probe_session import RosProbeSession

FAKE = '''
import json, pathlib, signal, sys, time
signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
r=pathlib.Path(sys.argv[1]); (r/'ready.json').write_text(json.dumps({'status':'ready'}))
n=0
while True:
 p=r/f'request_{n:03d}.json'
 if p.exists():
  q=json.loads(p.read_text())
  if q['operation']=='hang':
   time.sleep(5)
  else:
   t=r/f'result_{n:03d}.tmp'
   t.write_text(json.dumps({'status':'completed' if q['operation']!='fail' else 'failed','pid':__import__('os').getpid(),'operation':q['operation']}))
   t.rename(r/f'result_{n:03d}.json')
  n+=1
 time.sleep(.005)
'''


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)/'session'
        self.session = RosProbeSession(self.root, [sys.executable, '-c', FAKE, str(self.root)])

    def tearDown(self):
        self.session.close()
        self.assertIsNotNone(self.session.child.poll())
        self.tmp.cleanup()

    def test_reuses_ready_process_for_capture_motion_and_next_capture(self):
        results = [self.session.request({'operation': op}) for op in ('capture', 'motion', 'capture')]
        self.assertEqual(len({r['pid'] for r in results}), 1)
        self.assertEqual([r['operation'] for r in results], ['capture', 'motion', 'capture'])
        self.assertEqual(len(list(self.root.glob('request_*.json'))), 3)
        self.assertFalse(list(self.root.glob('*.tmp')))

    def test_failed_response_is_not_reported_as_success(self):
        with self.assertRaisesRegex(RuntimeError, 'ROS 操作失败'):
            self.session.request({'operation':'fail'})

    def test_timeout_remains_enforced_and_process_can_be_interrupted(self):
        with self.assertRaises(TimeoutError):
            self.session.request({'operation':'hang'}, timeout=.05)


if __name__ == '__main__':
    unittest.main(verbosity=2)
