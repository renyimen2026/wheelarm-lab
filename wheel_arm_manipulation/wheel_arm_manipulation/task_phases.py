"""Shared task-event phase intervals, using half-open ROS-time bounds."""

def parse_phases(events, task_id):
    intervals, active = [], None
    for event in events:
        if event["task_id"] != task_id:
            raise ValueError("阶段事件与原始任务 ID 不一致")
        if event["event"] == "phase_started":
            if active is not None:
                raise ValueError("阶段事件嵌套或缺少结束")
            active = (event["phase"], event["stamp_ns"])
        elif event["event"] == "phase_finished":
            if active is None or active[0] != event["phase"] or event["stamp_ns"] < active[1]:
                raise ValueError("阶段起止事件不匹配")
            if intervals and active[1] < intervals[-1][2]:
                raise ValueError("阶段时间区间重叠")
            intervals.append((active[0], active[1], event["stamp_ns"]))
            active = None
    if active is not None or not intervals:
        raise ValueError("阶段事件尚未完整结束")
    return intervals


def phase_at(stamp, intervals):
    for phase, start, end in intervals:
        if start <= stamp < end:
            return phase
    return "before_or_between_phases"
