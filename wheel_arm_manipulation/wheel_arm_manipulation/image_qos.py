"""Explicit image subscription profiles for controlled reliability comparisons."""

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


def image_qos(reliability):
    policies = {"best_effort": ReliabilityPolicy.BEST_EFFORT, "reliable": ReliabilityPolicy.RELIABLE}
    if reliability not in policies:
        raise ValueError("Reliability must be best_effort or reliable")
    return QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=5,
                      reliability=policies[reliability], durability=DurabilityPolicy.VOLATILE)
