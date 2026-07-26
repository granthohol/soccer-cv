# src/soccer_cv/cli.py
from __future__ import annotations

import argparse
import sys
from typing import Callable, List, Optional

from soccer_cv import (
    __version__,
    write_ball_path_2d_video,
    write_voronoi_2d_video,
    write_team_heatmaps_video,
    write_team_player_heatmap_grids,
    write_possession_2d_video,
    write_pass_network,
    write_team_shape_video,
    write_tracking_video,
    summarize_player_stats,
    write_side_by_side_video,
    write_video_with_image,
)


def _run_simple(fn: Callable[[str, str], None], args: argparse.Namespace) -> None:
    fn(args.source_video, args.target_video)
    print(f"Wrote: {args.target_video}")


def _add_simple_video_command(
    subparsers: "argparse._SubParsersAction",
    name: str,
    fn: Callable[[str, str], None],
    help_text: str,
) -> None:
    """Register a subcommand for the common (source_video, target_video) -> None shape."""
    p = subparsers.add_parser(name, help=help_text)
    p.add_argument("source_video", help="Path to input video")
    p.add_argument("target_video", help="Path to output video")
    p.set_defaults(func=lambda args, fn=fn: _run_simple(fn, args))


def _handle_team_shape(args: argparse.Namespace) -> None:
    write_team_shape_video(args.source_video, args.target_video, shape_every=args.shape_every)
    print(f"Wrote: {args.target_video}")


def _handle_possession(args: argparse.Namespace) -> None:
    write_possession_2d_video(args.source_video, args.target_video)
    print(f"Wrote: {args.target_video}")


def _handle_pass_network(args: argparse.Namespace) -> None:
    write_pass_network(
        args.source_video, args.output_path,
        min_presence_frames=args.min_presence_frames,
        confirm_frames=args.confirm_frames,
        radius_px=args.radius_px,
    )
    print(f"Wrote: {args.output_path}")


def _handle_heatmap_grids(args: argparse.Namespace) -> None:
    team0_path, team1_path = write_team_player_heatmap_grids(
        args.source_video, args.output_dir,
        grid_cols=args.grid_cols, normalize=args.normalize,
        cmap_name=args.cmap_name, alpha_max=args.alpha_max, blur_sigma=args.blur_sigma,
    )
    print(f"Wrote: {team0_path}")
    print(f"Wrote: {team1_path}")


def _handle_summarize(args: argparse.Namespace) -> None:
    summarize_player_stats(
        args.csv_path, output_csv=args.output_csv,
        speed_hi=args.speed_hi, speed_sprint=args.speed_sprint, accel_thr=args.accel_thr,
    )
    if args.output_csv:
        print(f"Wrote: {args.output_csv}")


def _handle_compare_side_by_side(args: argparse.Namespace) -> None:
    write_side_by_side_video(
        args.left_video, args.right_video, args.out_video,
        left_title=args.left_title, right_title=args.right_title,
        output_fps=args.output_fps, max_frames=args.max_frames,
    )
    print(f"Wrote: {args.out_video}")


def _handle_compare_with_image(args: argparse.Namespace) -> None:
    write_video_with_image(
        args.video_path, args.image_path, args.out_video,
        side=args.side, video_title=args.video_title, image_title=args.image_title,
        panel_width=args.panel_width, output_fps=args.output_fps, max_frames=args.max_frames,
    )
    print(f"Wrote: {args.out_video}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soccer-cv",
        description="Turn raw soccer broadcast footage into tracking data, tactical visuals, and CSV/PNG exports.",
    )
    parser.add_argument("--version", action="version", version=f"soccer-cv {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_simple_video_command(subparsers, "tracking", write_tracking_video,
        "Render a tracking video with per-player IDs/speeds (also writes a per-frame metrics CSV)")
    _add_simple_video_command(subparsers, "voronoi", write_voronoi_2d_video,
        "Render a team-control Voronoi diagram video")
    _add_simple_video_command(subparsers, "ball-path", write_ball_path_2d_video,
        "Render the ball's 2D path over time")
    _add_simple_video_command(subparsers, "heatmaps", write_team_heatmaps_video,
        "Render a cumulative team heatmap video (split screen)")

    p = subparsers.add_parser("team-shape", help="Render a continuous team-shape overlay video")
    p.add_argument("source_video", help="Path to input video")
    p.add_argument("target_video", help="Path to output video")
    p.add_argument("--shape-every", type=int, default=5,
                    help="Recompute the convex-hull team shape every N frames (default: 5)")
    p.set_defaults(func=_handle_team_shape)

    p = subparsers.add_parser("heatmap-grids", help="Generate per-player cumulative heatmap grid PNGs for both teams")
    p.add_argument("source_video", help="Path to input video")
    p.add_argument("output_dir", help="Directory to write output PNGs/CSV/npz into")
    p.add_argument("--grid-cols", type=int, default=4, help="Number of columns in the tiled grid (default: 4)")
    p.add_argument("--normalize", choices=["presence", "clip"], default="presence",
                    help="Heatmap normalization mode (default: presence)")
    p.add_argument("--cmap-name", default="JET", help="OpenCV colormap name (default: JET)")
    p.add_argument("--alpha-max", type=float, default=0.7, help="Max heatmap overlay opacity (default: 0.7)")
    p.add_argument("--blur-sigma", type=float, default=5.0, help="Gaussian blur sigma applied to heat grids (default: 5.0)")
    p.set_defaults(func=_handle_heatmap_grids)

    p = subparsers.add_parser("possession", help="Render a possession HUD video and classify pass/turnover events")
    p.add_argument("source_video", help="Path to input video")
    p.add_argument("target_video", help="Path to output video")
    p.set_defaults(func=_handle_possession)

    p = subparsers.add_parser("pass-network", help="Render a static pass-network PNG (nodes = players, edges = pass counts)")
    p.add_argument("source_video", help="Path to input video")
    p.add_argument("output_path", help="Path to output PNG")
    p.add_argument("--min-presence-frames", type=int, default=8,
                    help="Minimum visible frames for a player to get a node (default: 8)")
    p.add_argument("--confirm-frames", type=int, default=3,
                    help="Consecutive frames a new nearest player must hold before a possession change is confirmed (default: 3)")
    p.add_argument("--radius-px", type=float, default=100.0,
                    help="Max pitch-space distance for a player to be considered in possession of the ball (default: 100.0)")
    p.set_defaults(func=_handle_pass_network)

    p = subparsers.add_parser("summarize", help="Compute and print per-player stats from a tracking metrics CSV")
    p.add_argument("csv_path", help="Path to a tracking metrics CSV (from `soccer-cv tracking`)")
    p.add_argument("--output-csv", default=None, help="Optional path to also write the summary as a CSV")
    p.add_argument("--speed-hi", type=float, default=5.0, help="m/s threshold for 'high-intensity' running (default: 5.0)")
    p.add_argument("--speed-sprint", type=float, default=7.0, help="m/s threshold for 'sprint' (default: 7.0)")
    p.add_argument("--accel-thr", type=float, default=2.5, help="m/s^2 threshold for a high-acceleration event (default: 2.5)")
    p.set_defaults(func=_handle_summarize)

    p = subparsers.add_parser("compare-side-by-side", help="Stitch two videos side by side (e.g. input vs. output)")
    p.add_argument("left_video", help="Path to left video")
    p.add_argument("right_video", help="Path to right video")
    p.add_argument("out_video", help="Path to output video")
    p.add_argument("--left-title", default="Input", help="Label drawn on the left panel (default: Input)")
    p.add_argument("--right-title", default="Output", help="Label drawn on the right panel (default: Output)")
    p.add_argument("--output-fps", type=float, default=None, help="Override output FPS (default: left video's FPS)")
    p.add_argument("--max-frames", type=int, default=None, help="Cap total output frames (default: shorter of the two inputs)")
    p.set_defaults(func=_handle_compare_side_by_side)

    p = subparsers.add_parser("compare-with-image", help="Display a video next to a static image (e.g. a heatmap PNG)")
    p.add_argument("video_path", help="Path to input video")
    p.add_argument("image_path", help="Path to a static image (e.g. heatmap PNG)")
    p.add_argument("out_video", help="Path to output video")
    p.add_argument("--side", choices=["left", "right"], default="right",
                    help="Which side the image panel appears on (default: right)")
    p.add_argument("--video-title", default="Input", help="Label drawn on the video panel (default: Input)")
    p.add_argument("--image-title", default="Heatmap", help="Label drawn on the image panel (default: Heatmap)")
    p.add_argument("--panel-width", type=int, default=None, help="Override image panel width (default: scaled to video height)")
    p.add_argument("--output-fps", type=float, default=None, help="Override output FPS (default: source video's FPS)")
    p.add_argument("--max-frames", type=int, default=None, help="Cap total output frames (default: full video length)")
    p.set_defaults(func=_handle_compare_with_image)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
