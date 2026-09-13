"""Structure-first layout analysis for historical Chinese documents."""

from .layout import (
    Box,
    LayoutConfig,
    LayoutDocument,
    LayoutNode,
    RuleGraph,
    RuleSegment,
    detect_layout,
    detect_rule_graph,
    draw_layout,
    draw_rule_graph,
)

__all__ = [
    "Box",
    "LayoutConfig",
    "LayoutDocument",
    "LayoutNode",
    "RuleGraph",
    "RuleSegment",
    "detect_layout",
    "detect_rule_graph",
    "draw_layout",
    "draw_rule_graph",
]
__version__ = "1.1.0.dev0"
