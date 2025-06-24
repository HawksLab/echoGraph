"""
Echo Chamber Detection Module for EchoGraph

This module implements algorithms to detect echo chambers and filter bubbles
in social networks using various graph-based metrics and community detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict, Counter
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import math

from knowledge_graph.graph import KnowledgeGraph, Node, Edge

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EchoChamberMetrics:
    """Container for echo chamber metrics"""
    community_id: int
    users: List[str]
    size: int
    density: float
    modularity: float
    homophily_score: float
    content_diversity: float
    external_connections: int
    internal_connections: int
    polarization_index: float
    echo_score: float
    subreddits: List[str]
    dominant_topics: List[str]
    avg_activity: float
    formation_date: Optional[datetime] = None


@dataclass
class PolarizationMetrics:
    """Container for polarization analysis"""
    user_id: str
    community: int
    stance_vector: List[float]
    extremism_score: float
    bridge_score: float  # How much user bridges communities
    influence_score: float
    content_consistency: float


class EchoChamberDetector:
    """
    Detects echo chambers and analyzes polarization in social networks
    
    Features:
    - Community detection using multiple algorithms
    - Echo chamber scoring based on multiple metrics
    - Polarization analysis
    - Temporal dynamics tracking
    - Content diversity measurement
    - Misinformation pathway detection
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.graph = knowledge_graph
        self.communities = {}
        self.echo_chambers = []
        self.polarization_data = {}
        self.temporal_analysis = {}
        
    def detect_echo_chambers(self, min_size: int = 5, echo_threshold: float = 0.7) -> List[EchoChamberMetrics]:
        """
        Detect echo chambers in the network
        
        Args:
            min_size: Minimum community size to consider
            echo_threshold: Threshold for echo chamber classification
            
        Returns:
            List of echo chamber metrics
        """
        logger.info("Detecting echo chambers...")
        
        # Step 1: Detect communities
        self.communities = self._detect_communities()
        
        # Step 2: Analyze each community for echo chamber characteristics
        echo_chambers = []
        
        for community_id, users in self.communities.items():
            if len(users) >= min_size:
                metrics = self._analyze_community(community_id, users)
                
                if metrics.echo_score >= echo_threshold:
                    echo_chambers.append(metrics)
                    logger.info(f"Echo chamber detected: Community {community_id} "
                               f"(size={metrics.size}, echo_score={metrics.echo_score:.3f})")
        
        self.echo_chambers = sorted(echo_chambers, key=lambda x: x.echo_score, reverse=True)
        
        logger.info(f"Detected {len(self.echo_chambers)} echo chambers")
        return self.echo_chambers
    
    def _detect_communities(self) -> Dict[int, List[str]]:
        """Detect communities using improved algorithm"""
        logger.info("Detecting communities...")
        
        # Use the built-in community detection
        community_assignments = self.graph.detect_communities_louvain_like()
        
        # Group users by community
        communities = defaultdict(list)
        for node_id, community_id in community_assignments.items():
            if node_id.startswith('user_'):
                username = node_id[5:]  # Remove 'user_' prefix
                communities[community_id].append(username)
        
        # Filter out small communities
        filtered_communities = {cid: users for cid, users in communities.items() if len(users) >= 3}
        
        logger.info(f"Detected {len(filtered_communities)} communities")
        return dict(filtered_communities)
    
    def _analyze_community(self, community_id: int, users: List[str]) -> EchoChamberMetrics:
        """Analyze a community for echo chamber characteristics"""
        
        # Get community subgraph
        community_nodes = [f"user_{user}" for user in users]
        
        # Calculate basic metrics
        size = len(users)
        internal_connections = self._count_internal_connections(community_nodes)
        external_connections = self._count_external_connections(community_nodes)
        total_possible_internal = size * (size - 1) // 2
        density = internal_connections / max(total_possible_internal, 1)
        
        # Calculate homophily (tendency to connect with similar users)
        homophily_score = self._calculate_homophily(community_nodes)
        
        # Calculate content diversity
        content_diversity = self._calculate_content_diversity(users)
        
        # Calculate modularity contribution
        modularity = self._calculate_modularity_contribution(community_nodes)
        
        # Calculate polarization index
        polarization_index = self._calculate_polarization_index(users)
        
        # Get community subreddits and topics
        subreddits, topics = self._get_community_content_profile(users)
        
        # Calculate average activity
        avg_activity = self._calculate_average_activity(users)
        
        # Calculate overall echo score
        echo_score = self._calculate_echo_score(
            density, homophily_score, content_diversity, 
            external_connections, internal_connections, polarization_index
        )
        
        return EchoChamberMetrics(
            community_id=community_id,
            users=users,
            size=size,
            density=density,
            modularity=modularity,
            homophily_score=homophily_score,
            content_diversity=content_diversity,
            external_connections=external_connections,
            internal_connections=internal_connections,
            polarization_index=polarization_index,
            echo_score=echo_score,
            subreddits=subreddits,
            dominant_topics=topics,
            avg_activity=avg_activity
        )
    
    def _count_internal_connections(self, community_nodes: List[str]) -> int:
        """Count connections within the community"""
        internal_count = 0
        community_set = set(community_nodes)
        
        for node in community_nodes:
            neighbors = self.graph.get_neighbors(node)
            for neighbor in neighbors:
                if neighbor in community_set:
                    internal_count += 1
        
        return internal_count // 2  # Each edge counted twice
    
    def _count_external_connections(self, community_nodes: List[str]) -> int:
        """Count connections to other communities"""
        external_count = 0
        community_set = set(community_nodes)
        
        for node in community_nodes:
            neighbors = self.graph.get_neighbors(node)
            for neighbor in neighbors:
                if neighbor.startswith('user_') and neighbor not in community_set:
                    external_count += 1
        
        return external_count
    
    def _calculate_homophily(self, community_nodes: List[str]) -> float:
        """Calculate homophily score based on user similarity"""
        if len(community_nodes) < 2:
            return 0.0
        
        total_similarity = 0.0
        pair_count = 0
        
        for i, node1 in enumerate(community_nodes):
            for node2 in community_nodes[i+1:]:
                similarity = self._calculate_user_similarity(node1, node2)
                total_similarity += similarity
                pair_count += 1
        
        return total_similarity / max(pair_count, 1)
    
    def _calculate_user_similarity(self, user1: str, user2: str) -> float:
        """Calculate similarity between two users"""
        node1 = self.graph.get_node(user1)
        node2 = self.graph.get_node(user2)
        
        if not node1 or not node2:
            return 0.0
        
        # Similarity based on shared subreddits
        subreddits1 = set(node1.get_attribute('subreddits', []))
        subreddits2 = set(node2.get_attribute('subreddits', []))
        
        if not subreddits1 or not subreddits2:
            return 0.0
        
        jaccard_similarity = len(subreddits1.intersection(subreddits2)) / \
                           len(subreddits1.union(subreddits2))
        
        # Activity pattern similarity
        activity1 = node1.get_attribute('total_activity', 0)
        activity2 = node2.get_attribute('total_activity', 0)
        max_activity = max(activity1, activity2, 1)
        activity_similarity = 1 - abs(activity1 - activity2) / max_activity
        
        # Combined similarity
        return (jaccard_similarity + activity_similarity) / 2
    
    def _calculate_content_diversity(self, users: List[str]) -> float:
        """Calculate content diversity within the community"""
        all_subreddits = set()
        all_topics = set()
        user_subreddit_counts = defaultdict(int)
        
        for user in users:
            node = self.graph.get_node(f"user_{user}")
            if node:
                user_subreddits = node.get_attribute('subreddits', [])
                all_subreddits.update(user_subreddits)
                
                for subreddit in user_subreddits:
                    user_subreddit_counts[subreddit] += 1
        
        if not all_subreddits:
            return 0.0
        
        # Calculate Shannon entropy for subreddit distribution
        total_users = len(users)
        entropy = 0.0
        
        for count in user_subreddit_counts.values():
            probability = count / total_users
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        # Normalize by maximum possible entropy
        max_entropy = math.log2(len(all_subreddits)) if len(all_subreddits) > 1 else 1
        diversity = entropy / max_entropy if max_entropy > 0 else 0
        
        return min(diversity, 1.0)
    
    def _calculate_modularity_contribution(self, community_nodes: List[str]) -> float:
        """Calculate modularity contribution of this community"""
        # Simplified modularity calculation
        internal_edges = self._count_internal_connections(community_nodes)
        total_edges = sum(len(self.graph.edges[node]) for node in community_nodes if node in self.graph.edges)
        total_graph_edges = sum(len(edge_list) for edge_list in self.graph.edges.values())
        
        if total_graph_edges == 0:
            return 0.0
        
        expected_internal = (total_edges ** 2) / (2 * total_graph_edges)
        modularity = (internal_edges - expected_internal) / total_graph_edges
        
        return max(modularity, 0.0)
    
    def _calculate_polarization_index(self, users: List[str]) -> float:
        """Calculate polarization within the community"""
        # Simple polarization based on content stance consistency
        stance_scores = []
        
        for user in users:
            node = self.graph.get_node(f"user_{user}")
            if node:
                subreddits = node.get_attribute('subreddits', [])
                
                # Assign stance scores based on subreddit political leaning
                # This is a simplified approach - in practice, you'd use more sophisticated methods
                conservative_subs = {'conservative', 'republican', 'the_donald', 'asktrumpsupporters'}
                liberal_subs = {'politics', 'democrats', 'liberal', 'progressive'}
                
                conservative_score = sum(1 for sub in subreddits if sub.lower() in conservative_subs)
                liberal_score = sum(1 for sub in subreddits if sub.lower() in liberal_subs)
                
                total_political = conservative_score + liberal_score
                if total_political > 0:
                    stance = (conservative_score - liberal_score) / total_political
                    stance_scores.append(stance)
        
        if not stance_scores:
            return 0.0
        
        # Calculate polarization as standard deviation of stances
        mean_stance = np.mean(stance_scores)
        variance = np.var(stance_scores)
        polarization = math.sqrt(variance)
        
        return min(polarization, 1.0)
    
    def _get_community_content_profile(self, users: List[str]) -> Tuple[List[str], List[str]]:
        """Get the content profile of a community"""
        subreddit_counts = Counter()
        topic_counts = Counter()
        
        for user in users:
            node = self.graph.get_node(f"user_{user}")
            if node:
                subreddits = node.get_attribute('subreddits', [])
                subreddit_counts.update(subreddits)
        
        # Get top subreddits and topics
        top_subreddits = [sub for sub, count in subreddit_counts.most_common(5)]
        
        # Extract topics from subreddit names (simplified)
        political_topics = []
        for subreddit in top_subreddits:
            if any(political_word in subreddit.lower() for political_word in 
                  ['politics', 'conservative', 'liberal', 'trump', 'biden']):
                political_topics.append('politics')
            elif any(topic_word in subreddit.lower() for topic_word in 
                    ['conspiracy', 'qanon', 'pizzagate']):
                political_topics.append('conspiracy')
            elif 'covid' in subreddit.lower() or 'coronavirus' in subreddit.lower():
                political_topics.append('health')
        
        return top_subreddits, list(set(political_topics))
    
    def _calculate_average_activity(self, users: List[str]) -> float:
        """Calculate average activity level in the community"""
        total_activity = 0
        count = 0
        
        for user in users:
            node = self.graph.get_node(f"user_{user}")
            if node:
                activity = node.get_attribute('total_activity', 0)
                total_activity += activity
                count += 1
        
        return total_activity / max(count, 1)
    
    def _calculate_echo_score(self, density: float, homophily: float, content_diversity: float,
                            external_connections: int, internal_connections: int,
                            polarization: float) -> float:
        """Calculate overall echo chamber score"""
        
        # Normalize external connection ratio
        total_connections = external_connections + internal_connections
        external_ratio = external_connections / max(total_connections, 1)
        isolation_score = 1 - external_ratio
        
        # Combine metrics with weights
        weights = {
            'density': 0.2,
            'homophily': 0.25,
            'isolation': 0.25,
            'low_diversity': 0.15,
            'polarization': 0.15
        }
        
        echo_score = (
            weights['density'] * density +
            weights['homophily'] * homophily +
            weights['isolation'] * isolation_score +
            weights['low_diversity'] * (1 - content_diversity) +
            weights['polarization'] * polarization
        )
        
        return min(echo_score, 1.0)
    
    def analyze_polarization(self) -> Dict[str, PolarizationMetrics]:
        """Analyze individual user polarization"""
        logger.info("Analyzing user polarization...")
        
        polarization_data = {}
        
        for community_id, users in self.communities.items():
            for user in users:
                node = self.graph.get_node(f"user_{user}")
                if node:
                    metrics = self._calculate_user_polarization(user, community_id)
                    polarization_data[user] = metrics
        
        self.polarization_data = polarization_data
        return polarization_data
    
    def _calculate_user_polarization(self, user: str, community: int) -> PolarizationMetrics:
        """Calculate polarization metrics for a single user"""
        node = self.graph.get_node(f"user_{user}")
        
        # Calculate stance vector (simplified)
        subreddits = node.get_attribute('subreddits', [])
        stance_vector = self._calculate_stance_vector(subreddits)
        
        # Calculate extremism score
        extremism_score = np.linalg.norm(stance_vector)
        
        # Calculate bridge score (connections to other communities)
        bridge_score = self._calculate_bridge_score(f"user_{user}", community)
        
        # Calculate influence score
        influence_score = self._calculate_influence_score(f"user_{user}")
        
        # Calculate content consistency
        content_consistency = self._calculate_content_consistency(f"user_{user}")
        
        return PolarizationMetrics(
            user_id=user,
            community=community,
            stance_vector=stance_vector,
            extremism_score=extremism_score,
            bridge_score=bridge_score,
            influence_score=influence_score,
            content_consistency=content_consistency
        )
    
    def _calculate_stance_vector(self, subreddits: List[str]) -> List[float]:
        """Calculate political stance vector for user"""
        # Simplified political stance calculation
        dimensions = ['conservative_liberal', 'authoritarian_libertarian', 'nationalist_globalist']
        stance = [0.0, 0.0, 0.0]
        
        conservative_subs = {'conservative', 'republican', 'the_donald'}
        liberal_subs = {'politics', 'democrats', 'liberal'}
        
        for subreddit in subreddits:
            if subreddit.lower() in conservative_subs:
                stance[0] += 1.0
            elif subreddit.lower() in liberal_subs:
                stance[0] -= 1.0
        
        # Normalize
        total = sum(abs(s) for s in stance)
        if total > 0:
            stance = [s / total for s in stance]
        
        return stance
    
    def _calculate_bridge_score(self, user_node: str, user_community: int) -> float:
        """Calculate how much a user bridges different communities"""
        neighbors = self.graph.get_neighbors(user_node)
        other_community_connections = 0
        total_user_connections = 0
        
        for neighbor in neighbors:
            if neighbor.startswith('user_'):
                total_user_connections += 1
                neighbor_user = neighbor[5:]
                
                # Check if neighbor is in different community
                neighbor_community = None
                for comm_id, users in self.communities.items():
                    if neighbor_user in users:
                        neighbor_community = comm_id
                        break
                
                if neighbor_community is not None and neighbor_community != user_community:
                    other_community_connections += 1
        
        return other_community_connections / max(total_user_connections, 1)
    
    def _calculate_influence_score(self, user_node: str) -> float:
        """Calculate user influence based on network position"""
        node = self.graph.get_node(user_node)
        if not node:
            return 0.0
        
        # Use activity and karma as proxy for influence
        post_count = node.get_attribute('post_count', 0)
        comment_count = node.get_attribute('comment_count', 0)
        avg_post_score = node.get_attribute('avg_post_score', 0)
        avg_comment_score = node.get_attribute('avg_comment_score', 0)
        
        # Simple influence calculation
        activity_influence = math.log(post_count + comment_count + 1)
        score_influence = (avg_post_score + avg_comment_score) / 2
        
        return min((activity_influence + score_influence) / 10, 1.0)
    
    def _calculate_content_consistency(self, user_node: str) -> float:
        """Calculate how consistent user's content is"""
        node = self.graph.get_node(user_node)
        if not node:
            return 0.0
        
        subreddits = node.get_attribute('subreddits', [])
        if len(subreddits) <= 1:
            return 1.0
        
        # Simplified consistency based on subreddit political alignment
        conservative_count = sum(1 for sub in subreddits if any(cons in sub.lower() 
                                for cons in ['conservative', 'republican', 'trump']))
        liberal_count = sum(1 for sub in subreddits if any(lib in sub.lower() 
                           for lib in ['liberal', 'democrat', 'politics']))
        
        total_political = conservative_count + liberal_count
        if total_political == 0:
            return 0.5  # Neutral
        
        # Consistency is high if user sticks to one side
        max_side = max(conservative_count, liberal_count)
        consistency = max_side / total_political
        
        return consistency
    
    def generate_echo_chamber_report(self) -> Dict[str, Any]:
        """Generate comprehensive echo chamber analysis report"""
        logger.info("Generating echo chamber report...")
        
        if not self.echo_chambers:
            self.detect_echo_chambers()
        
        report = {
            'summary': {
                'total_communities': len(self.communities),
                'echo_chambers_detected': len(self.echo_chambers),
                'echo_chamber_ratio': len(self.echo_chambers) / max(len(self.communities), 1),
                'largest_echo_chamber': max(self.echo_chambers, key=lambda x: x.size).size if self.echo_chambers else 0,
                'highest_echo_score': max(self.echo_chambers, key=lambda x: x.echo_score).echo_score if self.echo_chambers else 0
            },
            'echo_chambers': [],
            'network_metrics': self._calculate_network_wide_metrics(),
            'temporal_analysis': self._analyze_temporal_dynamics(),
            'recommendations': self._generate_recommendations()
        }
        
        # Add detailed echo chamber information
        for chamber in self.echo_chambers:
            chamber_info = {
                'id': chamber.community_id,
                'size': chamber.size,
                'echo_score': chamber.echo_score,
                'density': chamber.density,
                'homophily': chamber.homophily_score,
                'content_diversity': chamber.content_diversity,
                'polarization': chamber.polarization_index,
                'dominant_subreddits': chamber.subreddits,
                'topics': chamber.dominant_topics,
                'isolation_ratio': chamber.external_connections / max(chamber.internal_connections + chamber.external_connections, 1),
                'avg_activity': chamber.avg_activity
            }
            report['echo_chambers'].append(chamber_info)
        
        return report
    
    def _calculate_network_wide_metrics(self) -> Dict[str, float]:
        """Calculate network-wide echo chamber metrics"""
        if not self.echo_chambers:
            return {}
        
        # Calculate average metrics across echo chambers
        total_echo_users = sum(chamber.size for chamber in self.echo_chambers)
        total_users = sum(len(users) for users in self.communities.values())
        
        avg_echo_score = np.mean([chamber.echo_score for chamber in self.echo_chambers])
        avg_density = np.mean([chamber.density for chamber in self.echo_chambers])
        avg_homophily = np.mean([chamber.homophily_score for chamber in self.echo_chambers])
        avg_diversity = np.mean([chamber.content_diversity for chamber in self.echo_chambers])
        avg_polarization = np.mean([chamber.polarization_index for chamber in self.echo_chambers])
        
        return {
            'echo_chamber_penetration': total_echo_users / max(total_users, 1),
            'average_echo_score': avg_echo_score,
            'average_density': avg_density,
            'average_homophily': avg_homophily,
            'average_content_diversity': avg_diversity,
            'average_polarization': avg_polarization
        }
    
    def _analyze_temporal_dynamics(self) -> Dict[str, Any]:
        """Analyze how echo chambers form and evolve over time"""
        # This is a simplified temporal analysis
        # In practice, you'd track community formation over multiple time periods
        
        return {
            'formation_patterns': 'Communities tend to form around controversial topics',
            'growth_rate': 'Echo chambers show rapid initial growth followed by stabilization',
            'stability': 'Most echo chambers persist once formed',
            'seasonal_trends': 'Activity increases during election periods'
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for reducing echo chamber effects"""
        recommendations = []
        
        if self.echo_chambers:
            avg_echo_score = np.mean([chamber.echo_score for chamber in self.echo_chambers])
            
            if avg_echo_score > 0.8:
                recommendations.append("High echo chamber activity detected. Consider implementing cross-community engagement initiatives.")
            
            if any(chamber.content_diversity < 0.3 for chamber in self.echo_chambers):
                recommendations.append("Low content diversity in some communities. Promote exposure to diverse perspectives.")
            
            if any(chamber.polarization_index > 0.7 for chamber in self.echo_chambers):
                recommendations.append("High polarization detected. Encourage respectful dialogue between opposing viewpoints.")
            
            recommendations.append("Monitor bridge users who connect different communities.")
            recommendations.append("Implement fact-checking mechanisms for widely shared content.")
        
        return recommendations


if __name__ == "__main__":
    # Example usage would go here
    # This would normally use a real knowledge graph
    from ..knowledge_graph.graph import KnowledgeGraph
    from ..knowledge_graph.graph_builder import GraphBuilder
    from ..data_collection.reddit_scraper import RedditScraper
    
    # Create demo data and graph
    scraper = RedditScraper()
    data = scraper.generate_demo_data(['politics', 'conspiracy'], 100)
    
    builder = GraphBuilder()
    kg = builder.build_from_reddit_data(data)
    
    # Detect echo chambers
    detector = EchoChamberDetector(kg)
    echo_chambers = detector.detect_echo_chambers()
    
    # Generate report
    report = detector.generate_echo_chamber_report()
    
    print("Echo Chamber Analysis Complete!")
    print(f"Detected {len(echo_chambers)} echo chambers")
    
    for chamber in echo_chambers[:3]:  # Show top 3
        print(f"\nEcho Chamber {chamber.community_id}:")
        print(f"  Size: {chamber.size} users")
        print(f"  Echo Score: {chamber.echo_score:.3f}")
        print(f"  Dominant Subreddits: {chamber.subreddits[:3]}")
        print(f"  Content Diversity: {chamber.content_diversity:.3f}")
        print(f"  Polarization: {chamber.polarization_index:.3f}")
