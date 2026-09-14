"""Structure-first layout analysis for historical Chinese documents."""

from .layout import (
    Box,
    LayoutConfig,
    LayoutDocument,
    LayoutNode,
    RegionCell,
    RegionGraph,
    RuleGraph,
    RuleSegment,
    detect_layout,
    detect_region_graph,
    detect_rule_graph,
    draw_layout,
    draw_region_graph,
    draw_rule_graph,
)

__all__ = [
    "Box",
    "LayoutConfig",
    "LayoutDocument",
    "LayoutNode",
    "RegionCell",
    "RegionGraph",
    "RuleGraph",
    "RuleSegment",
    "detect_layout",
    "detect_region_graph",
    "detect_rule_graph",
    "draw_layout",
    "draw_region_graph",
    "draw_rule_graph",
]
__version__ = "1.3.0.dev0"
