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
from .content_graph import (
    ContentEdge,
    ContentGraph,
    LaneObservation,
    StreamGroup,
    TextStream,
    detect_content_graph,
    draw_content_graph,
)

__all__ = [
    "Box",
    "ContentEdge",
    "ContentGraph",
    "LayoutConfig",
    "LayoutDocument",
    "LayoutNode",
    "LaneObservation",
    "StreamGroup",
    "RegionCell",
    "RegionGraph",
    "RuleGraph",
    "RuleSegment",
    "TextStream",
    "detect_content_graph",
    "draw_content_graph",
    "detect_layout",
    "detect_region_graph",
    "detect_rule_graph",
    "draw_layout",
    "draw_region_graph",
    "draw_rule_graph",
]
__version__ = "1.4.0.dev0"
