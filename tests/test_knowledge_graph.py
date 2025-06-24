"""
Unit tests for EchoGraph Knowledge Graph implementation
"""

import unittest
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.knowledge_graph.graph import KnowledgeGraph, Node, Edge


class TestKnowledgeGraph(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.kg = KnowledgeGraph()
        
        # Create test nodes
        self.user1 = Node("user1", "user", {"username": "alice", "karma": 1500})
        self.user2 = Node("user2", "user", {"username": "bob", "karma": 800})
        self.post1 = Node("post1", "post", {"title": "Test Post", "score": 45})
        
        # Add nodes to graph
        self.kg.add_node(self.user1)
        self.kg.add_node(self.user2)
        self.kg.add_node(self.post1)
    
    def test_node_creation(self):
        """Test node creation and attributes"""
        self.assertEqual(self.user1.id, "user1")
        self.assertEqual(self.user1.node_type, "user")
        self.assertEqual(self.user1.get_attribute("username"), "alice")
        self.assertEqual(self.user1.get_attribute("nonexistent", "default"), "default")
    
    def test_add_nodes(self):
        """Test adding nodes to the graph"""
        self.assertEqual(len(self.kg.nodes), 3)
        self.assertIn("user1", self.kg.nodes)
        self.assertIn("user2", self.kg.nodes)
        self.assertIn("post1", self.kg.nodes)
    
    def test_add_edges(self):
        """Test adding edges to the graph"""
        edge1 = Edge("user1", "user2", "follows", weight=1.0)
        edge2 = Edge("user1", "post1", "created", weight=1.0)
        
        self.kg.add_edge(edge1)
        self.kg.add_edge(edge2)
        
        self.assertEqual(len(self.kg.edges["user1"]), 2)
        self.assertEqual(len(self.kg.reverse_edges["user2"]), 1)
        self.assertEqual(len(self.kg.reverse_edges["post1"]), 1)
    
    def test_get_neighbors(self):
        """Test getting node neighbors"""
        edge1 = Edge("user1", "user2", "follows", weight=1.0)
        edge2 = Edge("user1", "post1", "created", weight=1.0)
        
        self.kg.add_edge(edge1)
        self.kg.add_edge(edge2)
        
        neighbors = self.kg.get_neighbors("user1")
        self.assertEqual(len(neighbors), 2)
        self.assertIn("user2", neighbors)
        self.assertIn("post1", neighbors)
        
        # Test edge type filtering
        follows_neighbors = self.kg.get_neighbors("user1", "follows")
        self.assertEqual(len(follows_neighbors), 1)
        self.assertIn("user2", follows_neighbors)
    
    def test_bfs(self):
        """Test breadth-first search"""
        # Create a small network
        user3 = Node("user3", "user", {"username": "charlie"})
        self.kg.add_node(user3)
        
        edge1 = Edge("user1", "user2", "follows", weight=1.0)
        edge2 = Edge("user2", "user3", "follows", weight=1.0)
        
        self.kg.add_edge(edge1)
        self.kg.add_edge(edge2)
        
        visited = self.kg.bfs("user1", max_depth=2)
        
        self.assertIn("user1", visited)
        self.assertIn("user2", visited)
        self.assertIn("user3", visited)
        self.assertEqual(visited["user1"], 0)
        self.assertEqual(visited["user2"], 1)
        self.assertEqual(visited["user3"], 2)
    
    def test_shortest_path(self):
        """Test shortest path algorithm"""
        user3 = Node("user3", "user", {"username": "charlie"})
        self.kg.add_node(user3)
        
        edge1 = Edge("user1", "user2", "follows", weight=1.0)
        edge2 = Edge("user2", "user3", "follows", weight=1.0)
        
        self.kg.add_edge(edge1)
        self.kg.add_edge(edge2)
        
        path = self.kg.shortest_path("user1", "user3")
        expected_path = ["user1", "user2", "user3"]
        
        self.assertEqual(path, expected_path)
        
        # Test path to non-existent node
        path_none = self.kg.shortest_path("user1", "nonexistent")
        self.assertIsNone(path_none)
    
    def test_connected_components(self):
        """Test connected components detection"""
        # Create disconnected component
        user3 = Node("user3", "user", {"username": "charlie"})
        user4 = Node("user4", "user", {"username": "david"})
        self.kg.add_node(user3)
        self.kg.add_node(user4)
        
        # Connect user1-user2 and user3-user4
        edge1 = Edge("user1", "user2", "follows")
        edge2 = Edge("user3", "user4", "follows")
        
        self.kg.add_edge(edge1)
        self.kg.add_edge(edge2)
        
        components = self.kg.get_connected_components()
        
        # Should have 2 components plus isolated post1
        self.assertEqual(len(components), 3)
        
        # Check component sizes
        component_sizes = [len(comp) for comp in components]
        component_sizes.sort()
        self.assertEqual(component_sizes, [1, 2, 2])
    
    def test_degree_calculation(self):
        """Test degree calculations"""
        edge1 = Edge("user1", "user2", "follows")
        edge2 = Edge("user1", "post1", "created")
        edge3 = Edge("user2", "user1", "follows")
        
        self.kg.add_edge(edge1)
        self.kg.add_edge(edge2)
        self.kg.add_edge(edge3)
        
        in_deg, out_deg, total_deg = self.kg.calculate_degree("user1")
        
        self.assertEqual(out_deg, 2)  # user1 -> user2, user1 -> post1
        self.assertEqual(in_deg, 1)   # user2 -> user1
        self.assertEqual(total_deg, 3)
    
    def test_community_detection(self):
        """Test simple community detection"""
        # Create small communities
        user3 = Node("user3", "user", {"username": "charlie"})
        user4 = Node("user4", "user", {"username": "david"})
        self.kg.add_node(user3)
        self.kg.add_node(user4)
        
        # Dense connections within communities
        edges = [
            Edge("user1", "user2", "follows"),
            Edge("user2", "user1", "follows"),
            Edge("user3", "user4", "follows"),
            Edge("user4", "user3", "follows"),
            Edge("user1", "user3", "follows")  # Bridge
        ]
        
        for edge in edges:
            self.kg.add_edge(edge)
        
        communities = self.kg.detect_communities_simple()
        
        # Should detect communities
        self.assertIsInstance(communities, dict)
        self.assertIn("user1", communities)
        self.assertIn("user2", communities)
    
    def test_graph_statistics(self):
        """Test graph statistics calculation"""
        edge1 = Edge("user1", "user2", "follows")
        self.kg.add_edge(edge1)
        
        stats = self.kg.get_graph_statistics()
        
        self.assertEqual(stats['num_nodes'], 3)
        self.assertEqual(stats['num_edges'], 1)
        self.assertIn('node_types', stats)
        self.assertIn('avg_degree', stats)
    
    def test_export_import(self):
        """Test graph export and import"""
        edge1 = Edge("user1", "user2", "follows", weight=2.0)
        self.kg.add_edge(edge1)
        
        # Export to dictionary
        graph_dict = self.kg.export_to_dict()
        
        self.assertIn('nodes', graph_dict)
        self.assertIn('edges', graph_dict)
        self.assertEqual(len(graph_dict['nodes']), 3)
        self.assertEqual(len(graph_dict['edges']), 1)
        
        # Import from dictionary
        new_kg = KnowledgeGraph.from_dict(graph_dict)
        
        self.assertEqual(len(new_kg.nodes), 3)
        self.assertEqual(len(new_kg.edges["user1"]), 1)
        self.assertEqual(new_kg.get_edge_weight("user1", "user2"), 2.0)


class TestGraphBuilder(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        from src.knowledge_graph.graph_builder import GraphBuilder
        from src.data_collection.reddit_scraper import RedditScraper
        
        self.builder = GraphBuilder()
        self.scraper = RedditScraper()
    
    def test_build_from_demo_data(self):
        """Test building graph from demo data"""
        # Generate demo data
        demo_data = self.scraper.generate_demo_data(['politics', 'conspiracy'], 20)
        
        # Build graph
        kg = self.builder.build_from_reddit_data(demo_data)
        
        # Check that graph was built
        self.assertIsNotNone(kg)
        self.assertGreater(len(kg.nodes), 0)
        
        # Check for different node types
        node_types = set(node.node_type for node in kg.nodes.values())
        self.assertIn('user', node_types)
        self.assertIn('post', node_types)
        self.assertIn('subreddit', node_types)


class TestEchoChamberDetector(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        from src.knowledge_graph.graph_builder import GraphBuilder
        from src.data_collection.reddit_scraper import RedditScraper
        from src.analysis.echo_chamber_detector import EchoChamberDetector
        
        # Create test graph
        scraper = RedditScraper()
        demo_data = scraper.generate_demo_data(['politics', 'conspiracy'], 30)
        
        builder = GraphBuilder()
        self.kg = builder.build_from_reddit_data(demo_data)
        
        self.detector = EchoChamberDetector(self.kg)
    
    def test_community_detection(self):
        """Test community detection"""
        communities = self.detector._detect_communities()
        
        self.assertIsInstance(communities, dict)
        self.assertGreater(len(communities), 0)
        
        # Check that communities contain users
        for community_id, users in communities.items():
            self.assertIsInstance(users, list)
            self.assertGreater(len(users), 0)
    
    def test_echo_chamber_detection(self):
        """Test echo chamber detection"""
        echo_chambers = self.detector.detect_echo_chambers(min_size=3, echo_threshold=0.5)
        
        self.assertIsInstance(echo_chambers, list)
        
        # If echo chambers found, check their structure
        for chamber in echo_chambers:
            self.assertGreater(chamber.size, 0)
            self.assertGreaterEqual(chamber.echo_score, 0.5)
            self.assertIsInstance(chamber.users, list)
    
    def test_report_generation(self):
        """Test report generation"""
        self.detector.detect_echo_chambers()
        report = self.detector.generate_echo_chamber_report()
        
        self.assertIn('summary', report)
        self.assertIn('echo_chambers', report)
        self.assertIn('network_metrics', report)
        self.assertIn('recommendations', report)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
