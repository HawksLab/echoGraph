"""
Simple Matplotlib-based Visualization for EchoGraph

This module provides reliable visualizations using matplotlib, seaborn, and networkx
for when web-based interactive visualizations are problematic.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import io
import base64
from datetime import datetime
import logging

from knowledge_graph.graph import KnowledgeGraph
from analysis.echo_chamber_detector import EchoChamberDetector, EchoChamberMetrics

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class SimpleGraphVisualizer:
    """
    Simple, reliable graph visualization using matplotlib
    
    Features:
    - Network plots with matplotlib/networkx
    - Echo chamber analysis charts
    - Statistical dashboards
    - Export to static images
    - Base64 encoding for web embedding
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.graph = knowledge_graph
        self.nx_graph = None
        self._prepare_networkx_graph()
    
    def _prepare_networkx_graph(self):
        """Convert knowledge graph to NetworkX for visualization"""
        self.nx_graph = nx.Graph()
        
        # Add nodes
        for node_id, node in self.graph.nodes.items():
            self.nx_graph.add_node(node_id, 
                                 node_type=node.node_type,
                                 **node.attributes)
        
        # Add edges
        for source, edges in self.graph.edges.items():
            for edge in edges:
                self.nx_graph.add_edge(source, edge.target,
                                     edge_type=edge.edge_type,
                                     weight=edge.weight)
        
        logger.info(f"NetworkX graph prepared: {len(self.nx_graph.nodes)} nodes, {len(self.nx_graph.edges)} edges")
    
    def create_network_plot(self, figsize=(12, 8), node_types=['user'], max_nodes=100):
        """Create a network visualization using matplotlib"""
        logger.info(f"Creating network plot for node types: {node_types}")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Filter nodes by type
        filtered_nodes = [n for n in self.nx_graph.nodes() 
                         if any(node_type in n for node_type in node_types)]
        
        # Limit nodes for performance
        if len(filtered_nodes) > max_nodes:
            filtered_nodes = filtered_nodes[:max_nodes]
            logger.info(f"Limited to {max_nodes} nodes for visualization")
        
        # Create subgraph
        subgraph = self.nx_graph.subgraph(filtered_nodes)
        
        # Calculate layout
        try:
            pos = nx.spring_layout(subgraph, k=1, iterations=50)
        except:
            pos = nx.random_layout(subgraph)
        
        # Color nodes by type
        node_colors = []
        for node in subgraph.nodes():
            if 'user_' in node:
                node_colors.append('#FF6B6B')  # Red for users
            elif 'post_' in node:
                node_colors.append('#4ECDC4')  # Teal for posts
            elif 'subreddit_' in node:
                node_colors.append('#45B7D1')  # Blue for subreddits
            elif 'topic_' in node:
                node_colors.append('#96CEB4')  # Green for topics
            else:
                node_colors.append('#FFEAA7')  # Yellow for others
        
        # Calculate node sizes based on degree
        degrees = dict(subgraph.degree())
        node_sizes = [min(degrees.get(node, 1) * 50, 500) for node in subgraph.nodes()]
        
        # Draw network
        nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.7, ax=ax)
        nx.draw_networkx_edges(subgraph, pos, alpha=0.3, width=0.5, ax=ax)
        
        # Add title and legend
        ax.set_title(f'Social Network Graph\n{len(subgraph.nodes)} nodes, {len(subgraph.edges)} edges', 
                    fontsize=14, fontweight='bold')
        
        # Create legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B', 
                      markersize=10, label='Users'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4ECDC4', 
                      markersize=10, label='Posts'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#45B7D1', 
                      markersize=10, label='Subreddits'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#96CEB4', 
                      markersize=10, label='Topics')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.axis('off')
        plt.tight_layout()
        
        return fig
    
    def create_echo_chamber_analysis(self, echo_chambers: List[EchoChamberMetrics], figsize=(15, 10)):
        """Create echo chamber analysis dashboard"""
        logger.info(f"Creating echo chamber analysis for {len(echo_chambers)} chambers")
        
        if not echo_chambers:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'No Echo Chambers Detected', 
                   ha='center', va='center', fontsize=16)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig
        
        # Prepare data
        df = pd.DataFrame([{
            'Chamber_ID': chamber.community_id,
            'Size': chamber.size,
            'Echo_Score': chamber.echo_score,
            'Density': chamber.density,
            'Content_Diversity': chamber.content_diversity,
            'Polarization': chamber.polarization_index,
            'Homophily': chamber.homophily_score
        } for chamber in echo_chambers])
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Echo Chamber Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Chamber sizes
        axes[0, 0].bar(range(len(df)), df['Size'], color='lightcoral')
        axes[0, 0].set_title('Echo Chamber Sizes')
        axes[0, 0].set_xlabel('Chamber Index')
        axes[0, 0].set_ylabel('Number of Users')
        
        # 2. Echo scores
        colors = ['red' if score > 0.7 else 'orange' if score > 0.5 else 'green' 
                 for score in df['Echo_Score']]
        axes[0, 1].bar(range(len(df)), df['Echo_Score'], color=colors)
        axes[0, 1].set_title('Echo Scores')
        axes[0, 1].set_xlabel('Chamber Index')
        axes[0, 1].set_ylabel('Echo Score')
        axes[0, 1].axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='High')
        axes[0, 1].axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Medium')
        axes[0, 1].legend()
        
        # 3. Size vs Echo Score scatter
        scatter = axes[0, 2].scatter(df['Size'], df['Echo_Score'], 
                                   c=df['Polarization'], cmap='Reds', s=100, alpha=0.7)
        axes[0, 2].set_title('Size vs Echo Score')
        axes[0, 2].set_xlabel('Chamber Size')
        axes[0, 2].set_ylabel('Echo Score')
        plt.colorbar(scatter, ax=axes[0, 2], label='Polarization')
        
        # 4. Content Diversity distribution
        axes[1, 0].hist(df['Content_Diversity'], bins=10, alpha=0.7, color='skyblue')
        axes[1, 0].set_title('Content Diversity Distribution')
        axes[1, 0].set_xlabel('Content Diversity Score')
        axes[1, 0].set_ylabel('Frequency')
        
        # 5. Metrics comparison
        metrics = ['Echo_Score', 'Density', 'Content_Diversity', 'Polarization']
        box_data = [df[metric].values for metric in metrics]
        axes[1, 1].boxplot(box_data, labels=metrics)
        axes[1, 1].set_title('Metrics Distribution')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Top chambers table
        axes[1, 2].axis('tight')
        axes[1, 2].axis('off')
        top_chambers = df.nlargest(5, 'Echo_Score')[['Chamber_ID', 'Size', 'Echo_Score']].round(3)
        table = axes[1, 2].table(cellText=top_chambers.values,
                               colLabels=top_chambers.columns,
                               cellLoc='center',
                               loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        axes[1, 2].set_title('Top Echo Chambers')
        
        plt.tight_layout()
        return fig
    
    def create_network_statistics_plot(self, figsize=(12, 8)):
        """Create network statistics visualization"""
        stats = self.graph.get_graph_statistics()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Network Statistics Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Node types distribution
        node_types = stats['node_types']
        axes[0, 0].pie(node_types.values(), labels=node_types.keys(), autopct='%1.1f%%')
        axes[0, 0].set_title('Node Types Distribution')
        
        # 2. Edge types distribution
        edge_types = stats['edge_types']
        axes[0, 1].bar(edge_types.keys(), edge_types.values())
        axes[0, 1].set_title('Edge Types Distribution')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Degree distribution
        degrees = [self.nx_graph.degree(n) for n in self.nx_graph.nodes()]
        axes[1, 0].hist(degrees, bins=20, alpha=0.7, color='lightgreen')
        axes[1, 0].set_title('Degree Distribution')
        axes[1, 0].set_xlabel('Node Degree')
        axes[1, 0].set_ylabel('Frequency')
        
        # 4. Summary statistics
        axes[1, 1].axis('off')
        summary_text = f"""
        Total Nodes: {stats['num_nodes']}
        Total Edges: {stats['num_edges']}
        Avg Degree: {np.mean(degrees):.2f}
        Max Degree: {max(degrees)}
        Connected Components: {stats['connected_components']}
        Graph Density: {nx.density(self.nx_graph):.4f}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
        axes[1, 1].set_title('Summary Statistics')
        
        plt.tight_layout()
        return fig
    
    def save_figure_as_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string for web embedding"""
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        plt.close(fig)  # Free memory
        return img_base64
    
    def get_visualization_html(self, fig) -> str:
        """Get matplotlib figure as HTML img tag"""
        img_base64 = self.save_figure_as_base64(fig)
        return f'<img src="data:image/png;base64,{img_base64}" style="max-width: 100%; height: auto;" alt="Visualization">'
    
    def create_all_visualizations(self) -> Dict[str, str]:
        """Create all visualizations and return as HTML"""
        visualizations = {}
        
        try:
            # Network plot
            network_fig = self.create_network_plot()
            visualizations['network'] = self.get_visualization_html(network_fig)
            logger.info("Network visualization created")
        except Exception as e:
            logger.error(f"Error creating network plot: {e}")
            visualizations['network'] = f'<div class="alert alert-danger">Error creating network plot: {e}</div>'
        
        try:
            # Network statistics
            stats_fig = self.create_network_statistics_plot()
            visualizations['statistics'] = self.get_visualization_html(stats_fig)
            logger.info("Statistics visualization created")
        except Exception as e:
            logger.error(f"Error creating statistics plot: {e}")
            visualizations['statistics'] = f'<div class="alert alert-danger">Error creating statistics plot: {e}</div>'
        
        return visualizations


if __name__ == "__main__":
    # Test the simple visualizer
    import sys
    import os
    sys.path.append('..')
    
    from data_collection.reddit_scraper import RedditScraper
    from knowledge_graph.graph_builder import GraphBuilder
    
    # Create test data
    scraper = RedditScraper()
    data = scraper.collect_subreddit_data(['python'], limit=20)
    
    # Build graph
    builder = GraphBuilder()
    graph = builder.build_from_reddit_data(data)
    
    # Create visualizations
    viz = SimpleGraphVisualizer(graph)
    
    # Test network plot
    fig = viz.create_network_plot()
    fig.savefig('test_network.png', dpi=150, bbox_inches='tight')
    print("Network plot saved as test_network.png")
    
    # Test statistics plot
    stats_fig = viz.create_network_statistics_plot()
    stats_fig.savefig('test_statistics.png', dpi=150, bbox_inches='tight')
    print("Statistics plot saved as test_statistics.png")
    
    plt.show()
