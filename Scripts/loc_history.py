#!/usr/bin/env python3
import argparse
import csv
import html
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def git(args):
    return subprocess.run(
        ["git", *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def git_text(args):
    return git(args).decode("utf-8", errors="replace")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a line-count-over-time chart from git history."
    )
    parser.add_argument(
        "--output",
        default="Analysis/loc_history.svg",
        help="SVG file to write. Default: Analysis/loc_history.svg",
    )
    parser.add_argument(
        "--csv",
        default="Analysis/loc_history.csv",
        help="CSV file to write. Use an empty value to skip CSV. Default: Analysis/loc_history.csv",
    )
    parser.add_argument(
        "--ext",
        action="append",
        default=None,
        help="Tracked file extension to count. Can be repeated. Default: .cs",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Path prefix to exclude. Can be repeated.",
    )
    parser.add_argument(
        "--title",
        default="Ramen C# LOC Over Time",
        help="Chart title.",
    )
    return parser.parse_args()


def commit_history():
    rows = []
    for line in git_text(["log", "--reverse", "--pretty=format:%H%x09%ct%x09%ad%x09%s", "--date=short"]).splitlines():
        commit, timestamp, date, subject = line.split("\t", 3)
        rows.append(
            {
                "commit": commit,
                "timestamp": int(timestamp),
                "date": date,
                "subject": subject,
            }
        )
    return rows


def included(path, extensions, excludes):
    return path.endswith(tuple(extensions)) and not any(path.startswith(prefix) for prefix in excludes)


def count_lines_at(commit, extensions, excludes):
    names = git(["ls-tree", "-r", "-z", "--name-only", commit]).split(b"\0")
    total = 0
    files = 0
    for raw_name in names:
        if not raw_name:
            continue
        path = raw_name.decode("utf-8", errors="surrogateescape")
        if not included(path, extensions, excludes):
            continue
        contents = git(["show", f"{commit}:{path}"])
        total += len(contents.splitlines())
        files += 1
    return total, files


def write_csv(path, rows):
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["date", "timestamp", "commit", "short_commit", "files", "loc", "subject"],
        )
        writer.writeheader()
        writer.writerows(rows)


def nice_ticks(maximum, count=5):
    if maximum <= 0:
        return [0]
    raw_step = maximum / count
    magnitude = 10 ** math.floor(math.log10(raw_step))
    normalized = raw_step / magnitude
    if normalized <= 1:
        step = magnitude
    elif normalized <= 2:
        step = 2 * magnitude
    elif normalized <= 5:
        step = 5 * magnitude
    else:
        step = 10 * magnitude
    return list(range(0, int(math.ceil(maximum / step) * step) + 1, int(step)))


def date_label(timestamp):
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")


def x_ticks(start, end, count=6):
    if start == end:
        return [start]
    return [round(start + (end - start) * i / (count - 1)) for i in range(count)]


def render_svg(path, rows, title):
    width = 1200
    height = 700
    left = 86
    right = 34
    top = 76
    bottom = 82
    chart_w = width - left - right
    chart_h = height - top - bottom

    min_t = min(row["timestamp"] for row in rows)
    max_t = max(row["timestamp"] for row in rows)
    max_loc = max(row["loc"] for row in rows)
    y_max = max(nice_ticks(max_loc)[-1], 1)

    def x(timestamp):
        if min_t == max_t:
            return left + chart_w / 2
        return left + ((timestamp - min_t) / (max_t - min_t)) * chart_w

    def y(loc):
        return top + chart_h - (loc / y_max) * chart_h

    points = " ".join(f"{x(row['timestamp']):.2f},{y(row['loc']):.2f}" for row in rows)
    y_grid = []
    for tick in nice_ticks(max_loc):
        ty = y(tick)
        y_grid.append(
            f'<line x1="{left}" y1="{ty:.2f}" x2="{width - right}" y2="{ty:.2f}" stroke="#e1e5eb" />'
            f'<text x="{left - 12}" y="{ty + 5:.2f}" text-anchor="end" class="tick">{tick:,}</text>'
        )

    x_grid = []
    for tick in x_ticks(min_t, max_t):
        tx = x(tick)
        x_grid.append(
            f'<line x1="{tx:.2f}" y1="{top}" x2="{tx:.2f}" y2="{height - bottom}" stroke="#eef1f4" />'
            f'<text x="{tx:.2f}" y="{height - 42}" text-anchor="middle" class="tick">{date_label(tick)}</text>'
        )

    first = rows[0]
    last = rows[-1]
    subtitle = (
        f"{len(rows):,} commits, {first['date']} to {last['date']}; "
        f"{last['loc']:,} LOC in {last['files']:,} files at {last['short_commit']}"
    )

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <style>
    .title {{ font: 700 28px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #111827; }}
    .subtitle {{ font: 15px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #4b5563; }}
    .label {{ font: 700 13px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #374151; }}
    .tick {{ font: 12px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #667085; }}
  </style>
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="{left}" y="38" class="title">{html.escape(title)}</text>
  <text x="{left}" y="62" class="subtitle">{html.escape(subtitle)}</text>
  {''.join(x_grid)}
  {''.join(y_grid)}
  <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#9ca3af" />
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#9ca3af" />
  <polyline fill="none" stroke="#2563eb" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="{points}" />
  <circle cx="{x(last['timestamp']):.2f}" cy="{y(last['loc']):.2f}" r="5" fill="#dc2626" />
  <text x="{width / 2:.2f}" y="{height - 14}" text-anchor="middle" class="label">Time</text>
  <text x="22" y="{top + chart_h / 2:.2f}" text-anchor="middle" class="label" transform="rotate(-90 22 {top + chart_h / 2:.2f})">Lines of code</text>
</svg>
''',
        encoding="utf-8",
    )


def main():
    args = parse_args()
    selected_extensions = args.ext or [".cs"]
    extensions = tuple(ext if ext.startswith(".") else f".{ext}" for ext in selected_extensions)
    rows = []
    commits = commit_history()
    for index, commit in enumerate(commits, 1):
        loc, files = count_lines_at(commit["commit"], extensions, tuple(args.exclude))
        rows.append(
            {
                **commit,
                "short_commit": commit["commit"][:7],
                "loc": loc,
                "files": files,
            }
        )
        if index == 1 or index == len(commits) or index % 25 == 0:
            print(f"{index}/{len(commits)} {commit['commit'][:7]} {loc} LOC", file=sys.stderr)

    write_csv(args.csv, rows)
    render_svg(args.output, rows, args.title)
    print(f"Wrote {args.output}")
    if args.csv:
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
