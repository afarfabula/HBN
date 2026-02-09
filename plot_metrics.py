import argparse
import csv
import os
import re


def _read_rows(metrics_path):
    with open(metrics_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = []
        for r in reader:
            rows.append(dict(r))
    return fieldnames, rows


def _as_int(x, default=None):
    if x is None:
        return default
    s = str(x).strip()
    if s == "":
        return default
    try:
        return int(float(s))
    except Exception:
        return default


def _as_float(x, default=None):
    if x is None:
        return default
    s = str(x).strip()
    if s == "":
        return default
    try:
        return float(s)
    except Exception:
        return default


def _detect_run_dir(run_dir, metrics_path):
    if metrics_path:
        metrics_path = os.path.abspath(metrics_path)
        return os.path.dirname(metrics_path), metrics_path

    run_dir = os.path.abspath(run_dir)
    metrics_path = os.path.join(run_dir, "metrics.csv")
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError("metrics.csv not found in {}".format(run_dir))
    return run_dir, metrics_path


def _find_latest_run_dir(runs_root):
    runs_root = os.path.abspath(runs_root)
    if not os.path.isdir(runs_root):
        raise FileNotFoundError("runs root not found: {}".format(runs_root))

    candidates = []
    for name in os.listdir(runs_root):
        p = os.path.join(runs_root, name)
        if not os.path.isdir(p):
            continue
        m = os.path.join(p, "metrics.csv")
        if os.path.isfile(m):
            candidates.append((os.path.getmtime(m), p))
    if not candidates:
        raise FileNotFoundError("no run dirs with metrics.csv under {}".format(runs_root))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _normalize_schema(fieldnames, rows):
    has_stage = "stage" in fieldnames
    has_epoch = "epoch" in fieldnames

    if not has_stage and "stage" not in fieldnames:
        pass

    out = []
    for r in rows:
        rr = dict(r)
        stage = _as_int(rr.get("stage"), default=None)
        if stage is None:
            stage = 1
        rr["stage"] = stage

        epoch = _as_int(rr.get("epoch"), default=None)
        if epoch is None:
            epoch = _as_int(rr.get("epoch"), default=None)
        if epoch is None and "epoch" not in rr:
            epoch = _as_int(rr.get("epoch"), default=None)
        if epoch is None and "epoch" not in fieldnames and "epoch" in rr:
            epoch = _as_int(rr["epoch"], default=None)
        if epoch is None:
            epoch = _as_int(rr.get("epoch"), default=0)
        rr["epoch"] = epoch

        rr["loss"] = _as_float(rr.get("loss"), default=None)
        rr["acc"] = _as_float(rr.get("acc"), default=None)
        rr["lr"] = _as_float(rr.get("lr"), default=None)
        rr["epsilon"] = _as_float(rr.get("epsilon"), default=None)
        rr["alpha"] = _as_float(rr.get("alpha"), default=None)
        rr["split"] = (rr.get("split") or "").strip()
        out.append(rr)

    return out


def _write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _stage_lengths(rows):
    stage_to_max_epoch = {}
    for r in rows:
        split = r.get("split") or ""
        if split == "stage_done":
            continue
        stage = int(r["stage"])
        epoch = int(r["epoch"])
        stage_to_max_epoch[stage] = max(stage_to_max_epoch.get(stage, -1), epoch)
    out = {}
    for stage, mx in stage_to_max_epoch.items():
        out[stage] = mx + 1
    return out


def _global_epoch(rows):
    lens = _stage_lengths(rows)
    stages = sorted(lens.keys())
    offsets = {}
    cur = 0
    for s in stages:
        offsets[s] = cur
        cur += lens[s]

    out = []
    for r in rows:
        rr = dict(r)
        s = int(rr["stage"])
        rr["global_epoch"] = offsets.get(s, 0) + int(rr["epoch"])
        out.append(rr)
    return out, offsets, lens


def _nice_split_order(splits):
    order = ["train", "test", "test_prev", "test_cand", "val", "stage_done"]
    key = {name: i for i, name in enumerate(order)}
    return sorted(splits, key=lambda s: (key.get(s, 999), s))


def _svg_escape(s):
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _polyline(points):
    return " ".join("{:.2f},{:.2f}".format(x, y) for x, y in points)


def _render_chart_svg(
    series_map,
    x_label,
    y_label,
    title,
    x_max,
    y_min,
    y_max,
    stage_boundaries,
    width=1100,
    height=560,
):
    pad_l = 60
    pad_r = 20
    pad_t = 45
    pad_b = 45

    plot_w = max(1, width - pad_l - pad_r)
    plot_h = max(1, height - pad_t - pad_b)

    def x_to_px(x):
        if x_max <= 0:
            return pad_l
        return pad_l + (float(x) / float(x_max)) * plot_w

    def y_to_px(y):
        if y_max <= y_min:
            return pad_t + plot_h / 2.0
        t = (float(y) - float(y_min)) / (float(y_max) - float(y_min))
        return pad_t + (1.0 - t) * plot_h

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    split_names = _nice_split_order(list(series_map.keys()))
    color_map = {}
    for i, s in enumerate(split_names):
        color_map[s] = colors[i % len(colors)]

    grid_lines = 5
    y_ticks = []
    if y_max > y_min:
        for i in range(grid_lines + 1):
            y = y_min + (y_max - y_min) * (i / float(grid_lines))
            y_ticks.append(y)

    svg = []
    svg.append(
        '<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'.format(
            w=width, h=height
        )
    )
    svg.append('<rect x="0" y="0" width="{w}" height="{h}" fill="white"/>'.format(w=width, h=height))

    svg.append(
        '<text x="{x}" y="{y}" font-family="sans-serif" font-size="16" font-weight="600">{t}</text>'.format(
            x=pad_l, y=24, t=_svg_escape(title)
        )
    )

    for y in y_ticks:
        py = y_to_px(y)
        svg.append(
            '<line x1="{x1:.2f}" y1="{y:.2f}" x2="{x2:.2f}" y2="{y:.2f}" stroke="#e6e6e6" stroke-width="1"/>'.format(
                x1=pad_l, x2=pad_l + plot_w, y=py
            )
        )
        svg.append(
            '<text x="{x:.2f}" y="{y:.2f}" font-family="sans-serif" font-size="12" fill="#444" text-anchor="end" dominant-baseline="middle">{v}</text>'.format(
                x=pad_l - 8, y=py, v=_svg_escape("{:.3g}".format(y))
            )
        )

    svg.append(
        '<line x1="{x:.2f}" y1="{y1:.2f}" x2="{x:.2f}" y2="{y2:.2f}" stroke="#333" stroke-width="1.2"/>'.format(
            x=pad_l, y1=pad_t, y2=pad_t + plot_h
        )
    )
    svg.append(
        '<line x1="{x1:.2f}" y1="{y:.2f}" x2="{x2:.2f}" y2="{y:.2f}" stroke="#333" stroke-width="1.2"/>'.format(
            x1=pad_l, x2=pad_l + plot_w, y=pad_t + plot_h
        )
    )

    for b in stage_boundaries:
        px = x_to_px(b)
        svg.append(
            '<line x1="{x:.2f}" y1="{y1:.2f}" x2="{x:.2f}" y2="{y2:.2f}" stroke="#999" stroke-width="1" stroke-dasharray="4 4"/>'.format(
                x=px, y1=pad_t, y2=pad_t + plot_h
            )
        )
        svg.append(
            '<text x="{x:.2f}" y="{y:.2f}" font-family="sans-serif" font-size="10" fill="#555" text-anchor="middle">{t}</text>'.format(
                x=px, y=pad_t - 8, t=_svg_escape('E={}'.format(int(b)))
            )
        )

    svg.append(
        '<text x="{x}" y="{y}" font-family="sans-serif" font-size="12" fill="#111" text-anchor="middle">{t}</text>'.format(
            x=pad_l + plot_w / 2.0, y=height - 14, t=_svg_escape(x_label)
        )
    )
    svg.append(
        '<text x="{x}" y="{y}" font-family="sans-serif" font-size="12" fill="#111" text-anchor="middle" transform="rotate(-90 {x} {y})">{t}</text>'.format(
            x=18, y=pad_t + plot_h / 2.0, t=_svg_escape(y_label)
        )
    )

    for split in split_names:
        pts = series_map.get(split) or []
        if len(pts) < 2:
            continue
        poly = []
        for x, y in pts:
            poly.append((x_to_px(x), y_to_px(y)))
        svg.append(
            '<polyline fill="none" stroke="{c}" stroke-width="2" points="{p}"/>'.format(
                c=color_map[split], p=_polyline(poly)
            )
        )

    legend_x = pad_l + 10
    legend_y = pad_t + 8
    legend_h = 18 * len(split_names) + 10
    legend_w = 200
    svg.append(
        '<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="white" stroke="#ddd"/>'.format(
            x=legend_x, y=legend_y, w=legend_w, h=legend_h
        )
    )
    for i, split in enumerate(split_names):
        y = legend_y + 22 + i * 18
        svg.append(
            '<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{c}" stroke-width="3"/>'.format(
                x1=legend_x + 10, x2=legend_x + 34, y=y, c=color_map[split]
            )
        )
        svg.append(
            '<text x="{x}" y="{y}" font-family="sans-serif" font-size="12" fill="#111" dominant-baseline="middle">{t}</text>'.format(
                x=legend_x + 42, y=y, t=_svg_escape(split)
            )
        )

    svg.append("</svg>")
    return "\n".join(svg)


def _series_from_rows(rows, x_key):
    series = {}
    for r in rows:
        split = r.get("split") or ""
        if split == "stage_done":
            continue
        x = _as_int(r.get(x_key), default=None)
        y_acc = _as_float(r.get("acc"), default=None)
        y_loss = _as_float(r.get("loss"), default=None)
        if x is None:
            continue
        if y_acc is not None:
            series.setdefault(("acc", split), []).append((x, y_acc))
        if y_loss is not None:
            series.setdefault(("loss", split), []).append((x, y_loss))

    out_acc = {}
    out_loss = {}
    for (k, split), pts in series.items():
        pts = sorted(pts, key=lambda t: t[0])
        if k == "acc":
            out_acc[split] = pts
        else:
            out_loss[split] = pts
    return out_acc, out_loss


def _write_svg(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(description="Split and visualize metrics.csv")
    parser.add_argument("--run-dir", default="", type=str, help="run directory containing metrics.csv")
    parser.add_argument("--metrics", default="", type=str, help="path to metrics.csv")
    parser.add_argument("--runs-root", default="./runs", type=str, help="root runs directory")
    args = parser.parse_args()

    run_dir = args.run_dir.strip()
    metrics_path = args.metrics.strip()
    if not run_dir and not metrics_path:
        run_dir = _find_latest_run_dir(args.runs_root)

    run_dir, metrics_path = _detect_run_dir(run_dir, metrics_path)
    fieldnames, raw_rows = _read_rows(metrics_path)
    rows = _normalize_schema(fieldnames, raw_rows)

    stages = sorted({int(r["stage"]) for r in rows})
    per_stage_fieldnames = list(fieldnames)
    if "global_epoch" not in per_stage_fieldnames:
        pass

    for s in stages:
        stage_rows = [r for r in rows if int(r["stage"]) == s]
        out_csv = os.path.join(run_dir, "metrics_stage{}.csv".format(s))
        _write_csv(out_csv, fieldnames, stage_rows)

    global_rows, offsets, lens = _global_epoch(rows)
    concat_fieldnames = list(fieldnames)
    if "global_epoch" not in concat_fieldnames:
        concat_fieldnames = ["global_epoch"] + concat_fieldnames
    out_concat = os.path.join(run_dir, "metrics_concat.csv")
    _write_csv(out_concat, concat_fieldnames, global_rows)

    boundaries = []
    for s in sorted(offsets.keys()):
        if offsets[s] > 0:
            boundaries.append(offsets[s])

    acc_series, loss_series = _series_from_rows(global_rows, "global_epoch")
    x_max = 0
    for pts in list(acc_series.values()) + list(loss_series.values()):
        if pts:
            x_max = max(x_max, max(x for x, _ in pts))

    acc_vals = [y for pts in acc_series.values() for _, y in pts]
    loss_vals = [y for pts in loss_series.values() for _, y in pts]
    acc_min, acc_max = (min(acc_vals), max(acc_vals)) if acc_vals else (0.0, 1.0)
    loss_min, loss_max = (min(loss_vals), max(loss_vals)) if loss_vals else (0.0, 1.0)

    acc_pad = max(1e-6, 0.05 * (acc_max - acc_min) if acc_max > acc_min else 1.0)
    loss_pad = max(1e-6, 0.05 * (loss_max - loss_min) if loss_max > loss_min else 1.0)

    acc_svg = _render_chart_svg(
        acc_series,
        x_label="global_epoch",
        y_label="acc",
        title="Accuracy (all stages concatenated)",
        x_max=x_max,
        y_min=acc_min - acc_pad,
        y_max=acc_max + acc_pad,
        stage_boundaries=boundaries,
        height=520,
    )
    loss_svg = _render_chart_svg(
        loss_series,
        x_label="global_epoch",
        y_label="loss",
        title="Loss (all stages concatenated)",
        x_max=x_max,
        y_min=loss_min - loss_pad,
        y_max=loss_max + loss_pad,
        stage_boundaries=boundaries,
        height=520,
    )

    out_acc_svg = os.path.join(run_dir, "curves_all_acc.svg")
    out_loss_svg = os.path.join(run_dir, "curves_all_loss.svg")
    _write_svg(out_acc_svg, acc_svg)
    _write_svg(out_loss_svg, loss_svg)

    for s in stages:
        stage_rows = [r for r in rows if int(r["stage"]) == s]
        acc_s, loss_s = _series_from_rows(stage_rows, "epoch")

        x_m = 0
        for pts in list(acc_s.values()) + list(loss_s.values()):
            if pts:
                x_m = max(x_m, max(x for x, _ in pts))

        acc_vals = [y for pts in acc_s.values() for _, y in pts]
        loss_vals = [y for pts in loss_s.values() for _, y in pts]
        a0, a1 = (min(acc_vals), max(acc_vals)) if acc_vals else (0.0, 1.0)
        l0, l1 = (min(loss_vals), max(loss_vals)) if loss_vals else (0.0, 1.0)
        ap = max(1e-6, 0.05 * (a1 - a0) if a1 > a0 else 1.0)
        lp = max(1e-6, 0.05 * (l1 - l0) if l1 > l0 else 1.0)

        acc_svg = _render_chart_svg(
            acc_s,
            x_label="epoch",
            y_label="acc",
            title="Accuracy (stage {})".format(s),
            x_max=x_m,
            y_min=a0 - ap,
            y_max=a1 + ap,
            stage_boundaries=[],
            height=520,
        )
        loss_svg = _render_chart_svg(
            loss_s,
            x_label="epoch",
            y_label="loss",
            title="Loss (stage {})".format(s),
            x_max=x_m,
            y_min=l0 - lp,
            y_max=l1 + lp,
            stage_boundaries=[],
            height=520,
        )

        _write_svg(os.path.join(run_dir, "curves_stage{}_acc.svg".format(s)), acc_svg)
        _write_svg(os.path.join(run_dir, "curves_stage{}_loss.svg".format(s)), loss_svg)

    print("run_dir:", run_dir)
    print("wrote:", os.path.join(run_dir, "metrics_concat.csv"))
    for s in stages:
        print("wrote:", os.path.join(run_dir, "metrics_stage{}.csv".format(s)))
    print("wrote:", os.path.join(run_dir, "curves_all_acc.svg"))
    print("wrote:", os.path.join(run_dir, "curves_all_loss.svg"))
    for s in stages:
        print("wrote:", os.path.join(run_dir, "curves_stage{}_acc.svg".format(s)))
        print("wrote:", os.path.join(run_dir, "curves_stage{}_loss.svg".format(s)))


if __name__ == "__main__":
    main()
