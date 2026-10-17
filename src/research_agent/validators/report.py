def validate_report(report, findings):
    hard, soft = [], []
    for p in all_points(report):
        p.finding_ids = [i for i in p.finding_ids if i in findings]   # 剔除非法 id
        if p.kind == "fact" and not p.finding_ids:
            soft.append((p, "fact 无证据"))
        if p.importance == "key" and not p.finding_ids:
            soft.append((p, "无关键证据支撑"))
        if p.kind != "inference" and not numbers_covered(p.text, [findings[i] for i in p.finding_ids]):
            soft.append((p, "数字未在证据中出现"))
    if not report.sections or any(not s.points for s in report.sections):
        hard.append("结构不完整")
    return hard, soft      # hard 非空 → 带错误信息重试;soft 存进 state,交给 generator 渲染