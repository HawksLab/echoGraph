"""
Graph Visualization Module for EchoGraph

This module provides interactive visualizations for social networks,
echo chambers, and misinformation pathways using various plotting libraries.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import colorsys
import math
import json
from datetime import datetime
import logging

from knowledge_graph.graph import KnowledgeGraph
from analysis.echo_chamber_detector import EchoChamberDetector, EchoChamberMetrics

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphVisualizer:
    """
    Interactive graph visualization for social network analysis
    
    Features:
    - Interactive network plots with Plotly
    - Echo chamber highlighting
    - Community detection visualization
    - Temporal analysis plots
    - Misinformation pathway tracing
    - Dashboard-style multi-panel views
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.graph = knowledge_graph
        self.nx_graph = None
        self.layout_cache = {}
        self.color_schemes = {
            'communities': px.colors.qualitative.Set3,
            'echo_chambers': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
            'polarization': ['#E74C3C', '#F39C12', '#F1C40F', '#2ECC71', '#3498DB']
        }
    
    def create_network_visualization(self, 
                                   node_types: List[str] = ['user'],
                                   edge_types: List[str] = None,
                                   layout: str = 'spring',
                                   show_labels: bool = True,
                                   node_size_attribute: str = 'total_activity',
                                   title: str = "Social Network Graph") -> go.Figure:
        """
        Create interactive network visualization
        
        Args:
            node_types: Types of nodes to include
            edge_types: Types of edges to include
            layout: Layout algorithm ('spring', 'circular', 'kamada_kawai')
            show_labels: Whether to show node labels
            node_size_attribute: Node attribute to use for sizing
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        logger.info(f"Creating network visualization with layout: {layout}")
        
        # Convert to NetworkX for layout algorithms
        nx_graph = self._convert_to_networkx(node_types, edge_types)
        
        # Calculate layout
        pos = self._calculate_layout(nx_graph, layout)
        
        # Prepare node data
        node_trace = self._create_node_trace(nx_graph, pos, node_size_attribute, show_labels)
        
        # Prepare edge data
        edge_trace = self._create_edge_trace(nx_graph, pos)
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace])
        
        fig.update_layout(
            title=title,
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Interactive Social Network - Hover for details, drag to pan, scroll to zoom",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color="gray", size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=700
        )
        
        return fig
    
    def create_echo_chamber_visualization(self, 
                                        echo_chambers: List[EchoChamberMetrics],
                                        layout: str = 'spring') -> go.Figure:
        """Create visualization highlighting echo chambers"""
        logger.info("Creating echo chamber visualization...")
        
        # Convert to NetworkX
        nx_graph = self._convert_to_networkx(['user'], ['similar_to', 'replied_to'])
        
        # Calculate layout
        pos = self._calculate_layout(nx_graph, layout)
        
        # Assign colors based on echo chambers
        node_colors = self._assign_echo_chamber_colors(echo_chambers)
        
        # Create traces for each echo chamber
        traces = []
        
        # Add edges
        edge_trace = self._create_edge_trace(nx_graph, pos, color='lightgray', width=0.5)
        traces.append(edge_trace)
        
        # Add nodes by echo chamber
        for i, chamber in enumerate(echo_chambers):
            chamber_nodes = [f"user_{user}" for user in chamber.users if f"user_{user}" in nx_graph.nodes()]
            
            if chamber_nodes:
                node_trace = self._create_chamber_node_trace(
                    chamber_nodes, pos, chamber, i
                )
                traces.append(node_trace)
        
        # Add non-echo chamber nodes
        all_echo_users = set()
        for chamber in echo_chambers:
            all_echo_users.update(f"user_{user}" for user in chamber.users)
        
        other_nodes = [node for node in nx_graph.nodes() if node not in all_echo_users]
        if other_nodes:
            other_trace = self._create_other_nodes_trace(other_nodes, pos)
            traces.append(other_trace)
        
        # Create figure
        fig = go.Figure(data=traces)
        
        fig.update_layout(
            title="Echo Chamber Analysis - Communities Highlighted by Color",
            titlefont_size=16,
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=700
        )
        
        return fig
    
    def create_echo_chamber_dashboard(self, 
                                    echo_chambers: List[EchoChamberMetrics],
                                    detector: EchoChamberDetector) -> go.Figure:
        """Create comprehensive dashboard for echo chamber analysis"""
        logger.info("Creating echo chamber dashboard...")
        
        # Create subplot structure
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Echo Chamber Sizes', 'Echo Chamber Scores', 'Content Diversity',
                'Network Overview', 'Polarization Analysis', 'Community Metrics'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "box"}, {"type": "scatter"}]
            ]
        )
        
        if echo_chambers:
            # 1. Echo Chamber Sizes
            chamber_ids = [f"Chamber {c.community_id}" for c in echo_chambers]
            sizes = [c.size for c in echo_chambers]
            
            fig.add_trace(
                go.Bar(x=chamber_ids, y=sizes, name="Size", marker_color='lightblue'),
                row=1, col=1
            )
            
            # 2. Echo Chamber Scores
            scores = [c.echo_score for c in echo_chambers]
            colors = ['red' if s > 0.7 else 'orange' if s > 0.5 else 'green' for s in scores]
            
            fig.add_trace(
                go.Bar(x=chamber_ids, y=scores, name="Echo Score", marker_color=colors),
                row=1, col=2
            )
            
            # 3. Content Diversity
            diversity = [c.content_diversity for c in echo_chambers]
            
            fig.add_trace(
                go.Bar(x=chamber_ids, y=diversity, name="Diversity", marker_color='purple'),
                row=1, col=3
            )
            
            # 4. Network Overview (scatter plot of chambers)
            fig.add_trace(
                go.Scatter(
                    x=sizes,
                    y=scores,
                    mode='markers',
                    marker=dict(
                        size=[s/2 for s in sizes],
                        color=diversity,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Diversity")
                    ),
                    text=chamber_ids,
                    name="Echo Chambers"
                ),
                row=2, col=1
            )
            
            # 5. Polarization Analysis
            polarization_scores = [c.polarization_index for c in echo_chambers]
            
            fig.add_trace(
                go.Box(y=polarization_scores, name="Polarization"),
                row=2, col=2
            )
            
            # 6. Community Metrics comparison
            metrics_df = pd.DataFrame({
                'Density': [c.density for c in echo_chambers],
                'Homophily': [c.homophily_score for c in echo_chambers],
                'Echo Score': [c.echo_score for c in echo_chambers]
            })
            
            for col_name in metrics_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(echo_chambers))),
                        y=metrics_df[col_name],
                        mode='lines+markers',
                        name=col_name
                    ),
                    row=2, col=3
                )
        
        # Update layout
        fig.update_layout(
            title_text="Echo Chamber Analysis Dashboard",
            height=800,
            showlegend=False
        )
        
        return fig
    
    def create_temporal_analysis(self, timeframe: str = '30d') -> go.Figure:
        """Create temporal analysis of network dynamics"""
        logger.info(f"Creating temporal analysis for {timeframe}")
        
        # Generate synthetic temporal data for demo
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        
        # Simulate metrics over time
        np.random.seed(42)
        base_trend = np.linspace(0.3, 0.7, 30)
        echo_scores = base_trend + np.random.normal(0, 0.05, 30)
        polarization = base_trend * 0.8 + np.random.normal(0, 0.03, 30)
        activity = 100 + 50 * np.sin(np.linspace(0, 4*np.pi, 30)) + np.random.normal(0, 10, 30)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Echo Chamber Formation Over Time',
                'Network Activity Levels',
                'Polarization Trends',
                'Community Growth'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        # Echo chamber formation
        fig.add_trace(
            go.Scatter(
                x=dates, y=echo_scores,
                mode='lines+markers',
                name='Echo Score',
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )
        
        # Network activity
        fig.add_trace(
            go.Scatter(
                x=dates, y=activity,
                mode='lines',
                name='Daily Posts',
                fill='tonexty',
                line=dict(color='blue')
            ),
            row=1, col=2
        )
        
        # Polarization trends
        fig.add_trace(
            go.Scatter(
                x=dates, y=polarization,
                mode='lines+markers',
                name='Polarization',
                line=dict(color='orange', width=2)
            ),
            row=2, col=1
        )
        
        # Community growth (cumulative)
        community_growth = np.cumsum(np.random.poisson(2, 30))
        fig.add_trace(
            go.Scatter(
                x=dates, y=community_growth,
                mode='lines',
                name='Total Communities',
                line=dict(color='green', width=3)
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Temporal Network Analysis",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_misinformation_pathway_viz(self, source_nodes: List[str], 
                                        max_depth: int = 3) -> go.Figure:
        """Visualize misinformation spread pathways"""
        logger.info("Creating misinformation pathway visualization...")
        
        # This is a simplified visualization
        # In practice, you'd trace actual content propagation
        
        # Create a subgraph showing information flow
        pathways = []
        for source in source_nodes:
            if f"user_{source}" in self.graph.nodes:
                visited = self.graph.bfs(f"user_{source}", max_depth)
                pathways.extend(list(visited.keys()))
        
        # Convert to NetworkX for visualization
        nx_graph = nx.DiGraph()
        
        # Add nodes and edges for pathways
        for node_id in set(pathways):
            if node_id in self.graph.nodes:
                nx_graph.add_node(node_id)
                
                # Add edges to show information flow
                neighbors = self.graph.get_neighbors(node_id)
                for neighbor in neighbors:
                    if neighbor in pathways:
                        nx_graph.add_edge(node_id, neighbor)
        
        # Calculate layout
        pos = nx.spring_layout(nx_graph, k=1, iterations=50)
        
        # Create visualization
        edge_trace = self._create_directed_edge_trace(nx_graph, pos)
        node_trace = self._create_pathway_node_trace(nx_graph, pos, source_nodes)
        
        fig = go.Figure(data=[edge_trace, node_trace])
        
        fig.update_layout(
            title="Information Propagation Pathways",
            titlefont_size=16,
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=600
        )
        
        return fig
    
    def _convert_to_networkx(self, node_types: List[str], edge_types: List[str] = None) -> nx.Graph:
        """Convert knowledge graph to NetworkX format"""
        nx_graph = nx.Graph()
        
        # Add nodes
        for node_id, node in self.graph.nodes.items():
            if node.node_type in node_types:
                nx_graph.add_node(node_id, **node.attributes)
        
        # Add edges
        for source, edge_list in self.graph.edges.items():
            if source in nx_graph.nodes():
                for edge in edge_list:
                    if (edge_types is None or edge.edge_type in edge_types) and edge.target in nx_graph.nodes():
                        nx_graph.add_edge(source, edge.target, 
                                        weight=edge.weight, 
                                        edge_type=edge.edge_type,
                                        **edge.attributes)
        
        return nx_graph
    
    def _calculate_layout(self, nx_graph: nx.Graph, layout: str) -> Dict[str, Tuple[float, float]]:
        """Calculate node positions using specified layout algorithm"""
        if layout in self.layout_cache:
            return self.layout_cache[layout]
        
        if layout == 'spring':
            pos = nx.spring_layout(nx_graph, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(nx_graph)
        elif layout == 'kamada_kawai':
            try:
                pos = nx.kamada_kawai_layout(nx_graph)
            except:
                pos = nx.spring_layout(nx_graph)
        else:
            pos = nx.spring_layout(nx_graph)
        
        self.layout_cache[layout] = pos
        return pos
    
    def _create_node_trace(self, nx_graph: nx.Graph, pos: Dict, 
                          size_attribute: str, show_labels: bool) -> go.Scatter:
        """Create node trace for plotting"""
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        node_sizes = []
        
        for node in nx_graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Get node info
            node_data = self.graph.get_node(node)
            if node_data:
                if node_data.node_type == 'user':
                    username = node_data.get_attribute('username', node)
                    activity = node_data.get_attribute(size_attribute, 1)
                    node_text.append(username if show_labels else '')
                    node_info.append(f"User: {username}<br>Activity: {activity}")
                    node_sizes.append(max(5, min(20, math.log(activity + 1) * 3)))
                else:
                    node_text.append(node if show_labels else '')
                    node_info.append(f"Node: {node}")
                    node_sizes.append(8)
            else:
                node_text.append('')
                node_info.append(f"Node: {node}")
                node_sizes.append(8)
        
        return go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text' if show_labels else 'markers',
            hoverinfo='text',
            text=node_text,
            hovertext=node_info,
            textposition="middle center",
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                size=node_sizes,
                color=node_sizes,
                colorbar=dict(
                    thickness=15,
                    len=0.5,
                    x=1.02,
                    title="Activity Level"
                ),
                line=dict(width=2, color='white')
            )
        )
    
    def _create_edge_trace(self, nx_graph: nx.Graph, pos: Dict, 
                          color: str = 'lightgray', width: float = 1.0) -> go.Scatter:
        """Create edge trace for plotting"""
        edge_x = []
        edge_y = []
        
        for edge in nx_graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        return go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=width, color=color),
            hoverinfo='none',
            mode='lines'
        )
    
    def _assign_echo_chamber_colors(self, echo_chambers: List[EchoChamberMetrics]) -> Dict[str, str]:
        """Assign colors to echo chambers"""
        colors = {}
        color_palette = self.color_schemes['echo_chambers']
        
        for i, chamber in enumerate(echo_chambers):
            color = color_palette[i % len(color_palette)]
            for user in chamber.users:
                colors[f"user_{user}"] = color
        
        return colors
    
    def _create_chamber_node_trace(self, chamber_nodes: List[str], pos: Dict,
                                  chamber: EchoChamberMetrics, chamber_idx: int) -> go.Scatter:
        """Create node trace for a specific echo chamber"""
        node_x = []
        node_y = []
        node_info = []
        
        for node in chamber_nodes:
            if node in pos:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                node_data = self.graph.get_node(node)
                username = node_data.get_attribute('username', node) if node_data else node
                node_info.append(
                    f"Echo Chamber {chamber.community_id}<br>"
                    f"User: {username}<br>"
                    f"Echo Score: {chamber.echo_score:.3f}<br>"
                    f"Chamber Size: {chamber.size}"
                )
        
        color = self.color_schemes['echo_chambers'][chamber_idx % len(self.color_schemes['echo_chambers'])]
        
        return go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            hovertext=node_info,
            marker=dict(
                size=12,
                color=color,
                line=dict(width=2, color='white')
            ),
            name=f"Echo Chamber {chamber.community_id} (Score: {chamber.echo_score:.2f})"
        )
    
    def _create_other_nodes_trace(self, other_nodes: List[str], pos: Dict) -> go.Scatter:
        """Create trace for non-echo chamber nodes"""
        node_x = []
        node_y = []
        node_info = []
        
        for node in other_nodes:
            if node in pos:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                node_data = self.graph.get_node(node)
                username = node_data.get_attribute('username', node) if node_data else node
                node_info.append(f"User: {username}<br>Status: Not in echo chamber")
        
        return go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            hovertext=node_info,
            marker=dict(
                size=8,
                color='lightgray',
                line=dict(width=1, color='white')
            ),
            name="Other Users"
        )
    
    def _create_directed_edge_trace(self, nx_graph: nx.DiGraph, pos: Dict) -> go.Scatter:
        """Create directed edge trace with arrows"""
        edge_x = []
        edge_y = []
        
        for edge in nx_graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        return go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='red'),
            hoverinfo='none',
            mode='lines',
            name='Information Flow'
        )
    
    def _create_pathway_node_trace(self, nx_graph: nx.Graph, pos: Dict,
                                  source_nodes: List[str]) -> go.Scatter:
        """Create node trace for misinformation pathways"""
        node_x = []
        node_y = []
        node_info = []
        node_colors = []
        
        for node in nx_graph.nodes():
            if node in pos:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                node_data = self.graph.get_node(node)
                username = node_data.get_attribute('username', node) if node_data else node
                
                is_source = any(f"user_{source}" == node for source in source_nodes)
                node_type = "Source" if is_source else "Receiver"
                node_colors.append('red' if is_source else 'lightblue')
                
                node_info.append(f"User: {username}<br>Role: {node_type}")
        
        return go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            hovertext=node_info,
            marker=dict(
                size=12,
                color=node_colors,
                line=dict(width=2, color='white')
            ),
            name='Users in Pathway'
        )
    
    def save_visualization(self, fig: go.Figure, filename: str, format: str = 'html') -> None:
        """Save visualization to file"""
        if format == 'html':
            # Use include_plotlyjs='inline' to ensure the HTML is standalone
            fig.write_html(filename, include_plotlyjs='inline', div_id="plotly-div")
        elif format == 'png':
            fig.write_image(filename)
        elif format == 'pdf':
            fig.write_image(filename)
        else:
            raise ValueError("Format must be 'html', 'png', or 'pdf'")
        
        logger.info(f"Visualization saved to {filename}")
    
    def get_visualization_html(self, fig: go.Figure) -> str:
        """Get visualization as HTML string for embedding"""
        # Create a simple, reliable HTML structure
        config = {
            'displayModeBar': True,
            'responsive': True,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'echograph_visualization',
                'height': 600,
                'width': 800,
                'scale': 1
            }
        }
        
        # Get figure as JSON
        fig_json = fig.to_json()
        
        # Create HTML with inline Plotly and direct plot creation
        html_content = f"""
        <div id="plotly-visualization" style="width:100%;height:600px;"></div>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <script type="text/javascript">
            (function() {{
                try {{
                    var figure = {fig_json};
                    var config = {json.dumps(config)};
                    
                    // Ensure the div exists before plotting
                    var plotDiv = document.getElementById('plotly-visualization');
                    if (plotDiv) {{
                        Plotly.newPlot('plotly-visualization', figure.data, figure.layout, config);
                        console.log('Plotly visualization created successfully');
                    }} else {{
                        console.error('Plot div not found');
                        setTimeout(function() {{
                            var retryDiv = document.getElementById('plotly-visualization');
                            if (retryDiv) {{
                                Plotly.newPlot('plotly-visualization', figure.data, figure.layout, config);
                                console.log('Plotly visualization created on retry');
                            }}
                        }}, 100);
                    }}
                }} catch (error) {{
                    console.error('Error creating Plotly visualization:', error);
                }}
            }})();
        </script>
        """
        
        return html_content


if __name__ == "__main__":
    # Example usage
    from ..knowledge_graph.graph_builder import GraphBuilder
    from ..data_collection.reddit_scraper import RedditScraper
    from ..analysis.echo_chamber_detector import EchoChamberDetector
    
    # Create demo data and graph
    scraper = RedditScraper()
    data = scraper.generate_demo_data(['politics', 'conspiracy'], 100)
    
    builder = GraphBuilder()
    kg = builder.build_from_reddit_data(data)
    
    # Detect echo chambers
    detector = EchoChamberDetector(kg)
    echo_chambers = detector.detect_echo_chambers()
    
    # Create visualizations
    viz = GraphVisualizer(kg)
    
    # Network overview
    network_fig = viz.create_network_visualization()
    viz.save_visualization(network_fig, 'network_overview.html')
    
    # Echo chamber visualization
    echo_fig = viz.create_echo_chamber_visualization(echo_chambers)
    viz.save_visualization(echo_fig, 'echo_chambers.html')
    
    # Dashboard
    dashboard_fig = viz.create_echo_chamber_dashboard(echo_chambers, detector)
    viz.save_visualization(dashboard_fig, 'echo_chamber_dashboard.html')
    
    print("Visualizations created successfully!")
    print("Files generated:")
    print("- network_overview.html")
    print("- echo_chambers.html") 
    print("- echo_chamber_dashboard.html")
