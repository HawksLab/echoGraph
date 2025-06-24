"""
EchoGraph Package Initialization

This package provides tools for social echo chamber analysis using knowledge graphs.
"""

__version__ = "1.0.0"
__author__ = "EchoGraph Team"
__email__ = "echograph@example.com"
__description__ = "Social Echo Chamber Analyzer using Knowledge Graphs"

# Import main classes for easy access
from .knowledge_graph.graph import KnowledgeGraph, Node, Edge
from .knowledge_graph.graph_builder import GraphBuilder
from .data_collection.reddit_scraper import RedditScraper
from .analysis.echo_chamber_detector import EchoChamberDetector
from .analysis.misinformation_detector import MisinformationDetector
from .visualization.graph_visualizer import GraphVisualizer

__all__ = [
    'KnowledgeGraph',
    'Node', 
    'Edge',
    'GraphBuilder',
    'RedditScraper',
    'EchoChamberDetector',
    'MisinformationDetector',
    'GraphVisualizer',
]
