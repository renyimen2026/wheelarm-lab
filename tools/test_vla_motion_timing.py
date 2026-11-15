"""ROS message classes only; mock the action transport, never create a ROS node."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import time
import math
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'wheel_arm_manipulation'))
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from wheel_arm_manipulation.arm_goal import ArmGoalClient
from wheel_arm_manipulation.planning import ArmModel
from nav_msgs.msg import Odometry
from execute_vla_short_motion import fresh_stationary_snapshot, require_single_execution_server, stationary_base


class DispatchTests(unittest.TestCase):
    def fixture(self):
        model = ArmModel(('j',), {'j':(-1,1)}, {'j':1}, {'j':1}, {})
        msg = JointState(name=['j'], position=[0.])
        node = SimpleNamespace(model=model, options=SimpleNamespace(wait=1), stopping=False,
                               execution_pending=False, snapshot=Mock(return_value=msg),
                               state=Mock(), valid_state=Mock(), wait=Mock(return_value=True))
        trajectory = RobotTrajectory()
        trajectory.joint_trajectory.joint_names=['j']
        trajectory.joint_trajectory.points=[JointTrajectoryPoint(positions=[0.])]
        return node, trajectory

    @patch('wheel_arm_manipulation.arm_goal.ActionClient')
    def test_stale_guard_aborts_after_checks_but_before_send(self, factory):
        node, trajectory = self.fixture()
        node.pre_execution_guard = Mock(side_effect=ValueError('stale observation'))
        with self.assertRaisesRegex(ValueError, 'stale'):
            ArmGoalClient.execute(node, trajectory, {'j':0.}, {'j':0.})
        node.valid_state.assert_called_once()
        factory.return_value.send_goal_async.assert_not_called()
        self.assertFalse(node.execution_pending)
        factory.return_value.destroy.assert_called_once()

    @patch('wheel_arm_manipulation.arm_goal.ActionClient')
    def test_existing_callers_without_guard_keep_action_path(self, factory):
        node, trajectory = self.fixture()
        factory.return_value.send_goal_async.return_value.result.return_value.accepted=False
        with self.assertRaisesRegex(RuntimeError, '拒绝执行'):
            ArmGoalClient.execute(node, trajectory, {'j':0.}, {'j':0.})
        factory.return_value.send_goal_async.assert_called_once()
        self.assertFalse(node.execution_pending)

    @patch('wheel_arm_manipulation.arm_goal.ActionClient')
    def test_specialized_snapshot_replaces_only_pre_execution_waits(self, factory):
        node, trajectory = self.fixture()
        node.pre_execution_snapshot = Mock(return_value=node.snapshot.return_value)
        factory.return_value.send_goal_async.return_value.result.return_value.accepted = False
        with self.assertRaisesRegex(RuntimeError, '拒绝执行'):
            ArmGoalClient.execute(node, trajectory, {'j':0.}, {'j':0.})
        self.assertEqual(node.pre_execution_snapshot.call_count, 2)
        node.snapshot.assert_not_called()


class ServerCountTests(unittest.TestCase):
    def test_requires_one_distinct_status_publisher(self):
        for count in (0, 2, 3):
            with self.assertRaises(RuntimeError):
                require_single_execution_server(SimpleNamespace(count_publishers=Mock(return_value=count)))
        node = SimpleNamespace(count_publishers=Mock(return_value=1))
        require_single_execution_server(node)
        node.count_publishers.assert_called_once_with('/execute_trajectory/_action/status')


class FreshSnapshotTests(unittest.TestCase):
    def fixture(self):
        message = JointState(name=['elbow_joint', 'left_finger_joint'], position=[0., .03], velocity=[0., 0.])
        odom = Odometry()
        odom.header.stamp.sec = 1
        return SimpleNamespace(snapshot=Mock(return_value=message), odom=odom,
            odom_received=time.monotonic(), model=SimpleNamespace(joints=tuple(message.name)),
            get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=1_100_000_000)))

    def test_fresh_stationary_feedback_returns_immediately(self):
        node = self.fixture()
        self.assertIs(fresh_stationary_snapshot(node), node.snapshot.return_value)
        node.snapshot.assert_called_once_with(settled=False)

    def test_moving_or_missing_joint_velocity_rejected(self):
        for velocity in ([.021, 0.], [0., .006], [math.nan, 0.], []):
            node = self.fixture()
            node.snapshot.return_value.velocity = velocity
            with self.assertRaises((ValueError, RuntimeError)):
                fresh_stationary_snapshot(node)

    @patch("execute_vla_short_motion.rclpy.spin_once")
    def test_stale_or_moving_base_rejected(self, _spin):
        for kind in ('wall', 'stamp', 'move', 'nan'):
            node = self.fixture()
            if kind == 'wall': node.odom_received -= 1
            if kind == 'stamp': node.odom.header.stamp.sec = 0
            if kind == 'move': node.odom.twist.twist.linear.x = .02
            if kind == 'nan': node.odom.twist.twist.angular.z = math.nan
            with self.assertRaises(RuntimeError):
                fresh_stationary_snapshot(node)

    @patch('execute_vla_short_motion.rclpy.spin_once')
    def test_small_future_stamp_is_waited_for_not_accepted(self, spin):
        node = self.fixture()
        node.odom.header.stamp.nanosec = 105_000_000  # 5ms ahead of the local clock.
        def catch_up(*args, **kwargs):
            node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=1_120_000_000))
        spin.side_effect = catch_up
        stationary_base(node)
        spin.assert_called_once()
        self.assertEqual(node.base_feedback_timing['recoveries'],1)
        self.assertAlmostEqual(node.base_feedback_timing['max_future_ms'],5.)

    @patch('execute_vla_short_motion.rclpy.spin_once')
    def test_queued_fresh_odom_can_replace_old_feedback(self, spin):
        node = self.fixture()
        node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=1_800_000_000))
        node.odom_received -= 1.
        def new_odom(*args, **kwargs):
            node.odom.header.stamp.nanosec = 750_000_000
            node.odom_received = time.monotonic()
        spin.side_effect = new_odom
        stationary_base(node)
        self.assertEqual(node.base_feedback_timing['recoveries'],1)

    @patch('execute_vla_short_motion.rclpy.spin_once')
    def test_persistent_future_feedback_still_fails_with_detailed_ages(self, spin):
        node = self.fixture()
        node.odom.header.stamp.nanosec = 105_000_000
        spin.side_effect = lambda *a, **k:time.sleep(.01)
        with self.assertRaisesRegex(RuntimeError, '100ms.*ROS年龄=-5.0ms'):
            stationary_base(node)
        self.assertEqual(node.base_feedback_timing['recoveries'],0)

    @patch('execute_vla_short_motion.rclpy.spin_once')
    def test_large_future_jump_and_moving_base_are_not_given_grace(self, spin):
        node = self.fixture()
        node.odom.header.stamp.nanosec = 250_000_000
        with self.assertRaisesRegex(RuntimeError, '时钟不一致'):
            stationary_base(node)
        spin.assert_not_called()
        node.odom.header.stamp.nanosec = 105_000_000
        node.odom.twist.twist.linear.x = .02
        with self.assertRaisesRegex(RuntimeError, '正在移动'):
            stationary_base(node)
        spin.assert_not_called()


if __name__=='__main__':
    unittest.main(verbosity=2)
