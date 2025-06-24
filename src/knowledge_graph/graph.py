"""
EchoGraph - Social Echo Chamber Analyzer
Custom Knowledge Graph Implementation

This module provides a from-scratch implementation of a knowledge graph
specifically designed for social media analysis and echo chamber detection.
"""

from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Any, Optional, Union
import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime
import math


@dataclass
class Node:
    """Represents a node in the knowledge graph"""
    id: str
    node_type: str  # 'user', 'post', 'subreddit', 'url', 'topic'
    attributes: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def add_attribute(self, key: str, value: Any):
        """Add an attribute to the node"""
        self.attributes[key] = value
    
    def get_attribute(self, key: str, default=None):
        """Get an attribute value"""
        return self.attributes.get(key, default)


@dataclass
class Edge:
    """Represents an edge in the knowledge graph"""
    source: str
    target: str
    edge_type: str  # 'follows', 'comments', 'shares', 'mentions', 'similar_to'
    weight: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def add_attribute(self, key: str, value: Any):
        """Add an attribute to the edge"""
        self.attributes[key] = value


class KnowledgeGraph:
    """
    Custom Knowledge Graph implementation for social media analysis
    
    Features:
    - Multi-type nodes (users, posts, subreddits, etc.)
    - Weighted and typed edges
    - Graph traversal algorithms
    - Community detection
    - Temporal analysis
    """
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, List[Edge]] = defaultdict(list)  # source -> [edges]
        self.reverse_edges: Dict[str, List[Edge]] = defaultdict(list)  # target -> [edges]
        self.node_types: Dict[str, Set[str]] = defaultdict(set)  # type -> {node_ids}
        self.edge_types: Dict[str, List[Edge]] = defaultdict(list)  # type -> [edges]
        
    def add_node(self, node: Node) -> None:
        """Add a node to the graph"""
        self.nodes[node.id] = node
        self.node_types[node.node_type].add(node.id)
    
    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph"""
        # Ensure both nodes exist
        if edge.source not in self.nodes or edge.target not in self.nodes:
            raise ValueError(f"Both source ({edge.source}) and target ({edge.target}) nodes must exist")
        
        self.edges[edge.source].append(edge)
        self.reverse_edges[edge.target].append(edge)
        self.edge_types[edge.edge_type].append(edge)
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID"""
        return self.nodes.get(node_id)
    
    def get_neighbors(self, node_id: str, edge_type: Optional[str] = None) -> List[str]:
        """Get neighboring nodes"""
        neighbors = []
        for edge in self.edges[node_id]:
            if edge_type is None or edge.edge_type == edge_type:
                neighbors.append(edge.target)
        return neighbors
    
    def get_incoming_neighbors(self, node_id: str, edge_type: Optional[str] = None) -> List[str]:
        """Get nodes that point to this node"""
        neighbors = []
        for edge in self.reverse_edges[node_id]:
            if edge_type is None or edge.edge_type == edge_type:
                neighbors.append(edge.source)
        return neighbors
    
    def get_edge_weight(self, source: str, target: str, edge_type: Optional[str] = None) -> float:
        """Get the weight of an edge between two nodes"""
        for edge in self.edges[source]:
            if edge.target == target and (edge_type is None or edge.edge_type == edge_type):
                return edge.weight
        return 0.0
    
    def bfs(self, start_node: str, max_depth: int = None) -> Dict[str, int]:
        """Breadth-First Search from a starting node"""
        if start_node not in self.nodes:
            return {}
        
        visited = {start_node: 0}
        queue = deque([(start_node, 0)])
        
        while queue:
            current, depth = queue.popleft()
            
            if max_depth is not None and depth >= max_depth:
                continue
                
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1))
        
        return visited
    
    def dfs(self, start_node: str, visited: Optional[Set[str]] = None) -> List[str]:
        """Depth-First Search from a starting node"""
        if visited is None:
            visited = set()
        
        if start_node in visited or start_node not in self.nodes:
            return []
        
        visited.add(start_node)
        path = [start_node]
        
        for neighbor in self.get_neighbors(start_node):
            path.extend(self.dfs(neighbor, visited))
        
        return path
    
    def shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find shortest path between two nodes using BFS"""
        if source not in self.nodes or target not in self.nodes:
            return None
        
        if source == target:
            return [source]
        
        queue = deque([(source, [source])])
        visited = {source}
        
        while queue:
            current, path = queue.popleft()
            
            for neighbor in self.get_neighbors(current):
                if neighbor == target:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def get_connected_components(self) -> List[Set[str]]:
        """Find all connected components in the graph"""
        visited = set()
        components = []
        
        for node_id in self.nodes:
            if node_id not in visited:
                component = set()
                stack = [node_id]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.add(current)
                        
                        # Add all neighbors (both directions)
                        stack.extend(self.get_neighbors(current))
                        stack.extend(self.get_incoming_neighbors(current))
                
                components.append(component)
        
        return components
    
    def calculate_degree(self, node_id: str) -> Tuple[int, int, int]:
        """Calculate in-degree, out-degree, and total degree"""
        if node_id not in self.nodes:
            return 0, 0, 0
        
        out_degree = len(self.edges[node_id])
        in_degree = len(self.reverse_edges[node_id])
        total_degree = in_degree + out_degree
        
        return in_degree, out_degree, total_degree
    
    def get_node_clustering_coefficient(self, node_id: str) -> float:
        """Calculate clustering coefficient for a node"""
        neighbors = set(self.get_neighbors(node_id))
        if len(neighbors) < 2:
            return 0.0
        
        # Count edges between neighbors
        edges_between_neighbors = 0
        for neighbor1 in neighbors:
            for neighbor2 in neighbors:
                if neighbor1 != neighbor2 and self.get_edge_weight(neighbor1, neighbor2) > 0:
                    edges_between_neighbors += 1
        
        # Calculate clustering coefficient
        possible_edges = len(neighbors) * (len(neighbors) - 1)
        return edges_between_neighbors / possible_edges if possible_edges > 0 else 0.0
    
    def detect_communities_simple(self) -> Dict[str, int]:
        """Simple community detection using connected components"""
        components = self.get_connected_components()
        community_map = {}
        
        for i, component in enumerate(components):
            for node_id in component:
                community_map[node_id] = i
        
        return community_map
    
    def detect_communities_louvain_like(self, resolution: float = 1.0) -> Dict[str, int]:
        """
        Simplified Louvain-like community detection algorithm
        """
        # Initialize each node in its own community
        communities = {node_id: i for i, node_id in enumerate(self.nodes.keys())}
        improved = True
        
        while improved:
            improved = False
            
            for node_id in self.nodes:
                current_community = communities[node_id]
                best_community = current_community
                best_gain = 0.0
                
                # Check neighboring communities
                neighbor_communities = set()
                for neighbor in self.get_neighbors(node_id):
                    neighbor_communities.add(communities[neighbor])
                
                for community in neighbor_communities:
                    if community != current_community:
                        # Calculate modularity gain (simplified)
                        gain = self._calculate_modularity_gain(node_id, community, communities)
                        if gain > best_gain:
                            best_gain = gain
                            best_community = community
                
                if best_community != current_community:
                    communities[node_id] = best_community
                    improved = True
        
        # Renumber communities to be consecutive
        unique_communities = list(set(communities.values()))
        community_mapping = {old: new for new, old in enumerate(unique_communities)}
        
        return {node_id: community_mapping[comm] for node_id, comm in communities.items()}
    
    def _calculate_modularity_gain(self, node_id: str, target_community: int, communities: Dict[str, int]) -> float:
        """Calculate modularity gain for moving a node to a target community"""
        # Simplified modularity calculation
        current_community = communities[node_id]
        
        # Count edges to target community vs current community
        edges_to_target = 0
        edges_to_current = 0
        
        for neighbor in self.get_neighbors(node_id):
            if communities[neighbor] == target_community:
                edges_to_target += 1
            elif communities[neighbor] == current_community:
                edges_to_current += 1
        
        return edges_to_target - edges_to_current
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        stats = {
            'num_nodes': len(self.nodes),
            'num_edges': sum(len(edge_list) for edge_list in self.edges.values()),
            'node_types': {node_type: len(nodes) for node_type, nodes in self.node_types.items()},
            'edge_types': {edge_type: len(edges) for edge_type, edges in self.edge_types.items()},
            'connected_components': len(self.get_connected_components()),
        }
        
        # Calculate degree statistics
        degrees = []
        for node_id in self.nodes:
            _, _, total_degree = self.calculate_degree(node_id)
            degrees.append(total_degree)
        
        if degrees:
            stats['avg_degree'] = sum(degrees) / len(degrees)
            stats['max_degree'] = max(degrees)
            stats['min_degree'] = min(degrees)
        
        return stats
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export graph to dictionary format"""
        return {
            'nodes': {
                node_id: {
                    'node_type': node.node_type,
                    'attributes': node.attributes,
                    'timestamp': node.timestamp.isoformat() if node.timestamp else None
                }
                for node_id, node in self.nodes.items()
            },
            'edges': [
                {
                    'source': edge.source,
                    'target': edge.target,
                    'edge_type': edge.edge_type,
                    'weight': edge.weight,
                    'attributes': edge.attributes,
                    'timestamp': edge.timestamp.isoformat() if edge.timestamp else None
                }
                for edge_list in self.edges.values()
                for edge in edge_list
            ]
        }
    
    def save_to_file(self, filepath: str, format: str = 'pickle') -> None:
        """Save graph to file"""
        if format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
        elif format == 'json':
            with open(filepath, 'w') as f:
                json.dump(self.export_to_dict(), f, indent=2)
        else:
            raise ValueError("Format must be 'pickle' or 'json'")
    
    @classmethod
    def load_from_file(cls, filepath: str, format: str = 'pickle') -> 'KnowledgeGraph':
        """Load graph from file"""
        if format == 'pickle':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif format == 'json':
            with open(filepath, 'r') as f:
                data = json.load(f)
            return cls.from_dict(data)
        else:
            raise ValueError("Format must be 'pickle' or 'json'")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeGraph':
        """Create graph from dictionary"""
        graph = cls()
        
        # Add nodes
        for node_id, node_data in data['nodes'].items():
            timestamp = None
            if node_data.get('timestamp'):
                timestamp = datetime.fromisoformat(node_data['timestamp'])
            
            node = Node(
                id=node_id,
                node_type=node_data['node_type'],
                attributes=node_data.get('attributes', {}),
                timestamp=timestamp
            )
            graph.add_node(node)
        
        # Add edges
        for edge_data in data['edges']:
            timestamp = None
            if edge_data.get('timestamp'):
                timestamp = datetime.fromisoformat(edge_data['timestamp'])
            
            edge = Edge(
                source=edge_data['source'],
                target=edge_data['target'],
                edge_type=edge_data['edge_type'],
                weight=edge_data.get('weight', 1.0),
                attributes=edge_data.get('attributes', {}),
                timestamp=timestamp
            )
            graph.add_edge(edge)
        
        return graph


if __name__ == "__main__":
    # Example usage
    kg = KnowledgeGraph()
    
    # Add some nodes
    user1 = Node("user1", "user", {"username": "alice", "karma": 1500})
    user2 = Node("user2", "user", {"username": "bob", "karma": 800})
    post1 = Node("post1", "post", {"title": "Sample Post", "upvotes": 45})
    
    kg.add_node(user1)
    kg.add_node(user2)
    kg.add_node(post1)
    
    # Add some edges
    follows_edge = Edge("user1", "user2", "follows", weight=1.0)
    comments_edge = Edge("user1", "post1", "comments", weight=1.0)
    
    kg.add_edge(follows_edge)
    kg.add_edge(comments_edge)
    
    # Test some operations
    print("Graph Statistics:", kg.get_graph_statistics())
    print("User1 neighbors:", kg.get_neighbors("user1"))
    print("Shortest path user1 -> post1:", kg.shortest_path("user1", "post1"))
