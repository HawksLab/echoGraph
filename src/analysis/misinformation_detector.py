"""
Misinformation Detection Module for EchoGraph

This module implements algorithms to detect and trace misinformation
pathways in social networks using content analysis and propagation patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import re
import math
from dataclasses import dataclass
import logging

from knowledge_graph.graph import KnowledgeGraph

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MisinformationCluster:
    """Container for misinformation cluster data"""
    cluster_id: int
    content_items: List[str]
    propagators: List[str]
    origin_users: List[str]
    spread_pattern: Dict[str, Any]
    similarity_score: float
    velocity: float  # Speed of spread
    reach: int  # Number of unique users reached
    coordination_score: float
    content_type: str
    detected_at: datetime


@dataclass
class PropagationPath:
    """Container for information propagation pathway"""
    source_user: str
    target_user: str
    content_id: str
    timestamp: datetime
    platform: str
    interaction_type: str
    delay_minutes: int
    influence_weight: float


class MisinformationDetector:
    """
    Detects coordinated misinformation campaigns and analyzes spread patterns
    
    Features:
    - Content similarity analysis for detecting coordinated messaging
    - Propagation pathway tracing
    - Bot-like behavior detection
    - Temporal pattern analysis
    - Network anomaly detection
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.graph = knowledge_graph
        self.content_clusters = []
        self.propagation_paths = []
        self.suspicious_users = set()
        self.coordination_networks = {}
    
    def detect_misinformation_campaigns(self, 
                                      similarity_threshold: float = 0.8,
                                      min_cluster_size: int = 5,
                                      velocity_threshold: float = 0.5) -> List[MisinformationCluster]:
        """
        Detect coordinated misinformation campaigns
        
        Args:
            similarity_threshold: Minimum content similarity for clustering
            min_cluster_size: Minimum number of items in a cluster
            velocity_threshold: Minimum spread velocity to consider suspicious
            
        Returns:
            List of detected misinformation clusters
        """
        logger.info("Detecting misinformation campaigns...")
        
        # Step 1: Content similarity analysis
        content_clusters = self._detect_similar_content_clusters(similarity_threshold, min_cluster_size)
        
        # Step 2: Analyze propagation patterns
        for cluster in content_clusters:
            self._analyze_cluster_propagation(cluster)
        
        # Step 3: Detect coordination patterns
        suspicious_clusters = []
        for cluster in content_clusters:
            if self._is_cluster_suspicious(cluster, velocity_threshold):
                suspicious_clusters.append(cluster)
        
        self.content_clusters = suspicious_clusters
        logger.info(f"Detected {len(suspicious_clusters)} suspicious content clusters")
        
        return suspicious_clusters
    
    def _detect_similar_content_clusters(self, similarity_threshold: float, 
                                       min_cluster_size: int) -> List[MisinformationCluster]:
        """Detect clusters of similar content"""
        logger.info("Analyzing content similarity...")
        
        # Get all posts and their content
        posts_content = {}
        posts_metadata = {}
        
        for node_id, node in self.graph.nodes.items():
            if node.node_type == 'post':
                title = node.get_attribute('title', '')
                content = node.get_attribute('selftext', '')
                combined_text = f"{title} {content}".lower()
                
                posts_content[node_id] = combined_text
                posts_metadata[node_id] = {
                    'author': node.get_attribute('author'),
                    'subreddit': node.get_attribute('subreddit'),
                    'score': node.get_attribute('score', 0),
                    'timestamp': node.timestamp,
                    'title': title
                }
        
        # Calculate content similarity matrix
        similarity_matrix = self._calculate_content_similarity_matrix(posts_content)
        
        # Cluster similar content
        clusters = self._cluster_similar_content(similarity_matrix, similarity_threshold, min_cluster_size)
        
        # Convert to MisinformationCluster objects
        misinformation_clusters = []
        for i, cluster_posts in enumerate(clusters):
            if len(cluster_posts) >= min_cluster_size:
                
                # Get cluster metadata
                cluster_authors = [posts_metadata[post]['author'] for post in cluster_posts]
                cluster_timestamps = [posts_metadata[post]['timestamp'] for post in cluster_posts if posts_metadata[post]['timestamp']]
                
                # Calculate similarity score
                cluster_similarities = []
                for j, post1 in enumerate(cluster_posts):
                    for post2 in cluster_posts[j+1:]:
                        if post1 in similarity_matrix and post2 in similarity_matrix[post1]:
                            cluster_similarities.append(similarity_matrix[post1][post2])
                
                avg_similarity = np.mean(cluster_similarities) if cluster_similarities else 0
                
                # Create cluster
                cluster = MisinformationCluster(
                    cluster_id=i,
                    content_items=cluster_posts,
                    propagators=list(set(cluster_authors)),
                    origin_users=[],  # Will be filled later
                    spread_pattern={},
                    similarity_score=avg_similarity,
                    velocity=0.0,  # Will be calculated
                    reach=len(set(cluster_authors)),
                    coordination_score=0.0,  # Will be calculated
                    content_type='text',
                    detected_at=datetime.now()
                )
                
                misinformation_clusters.append(cluster)
        
        return misinformation_clusters
    
    def _calculate_content_similarity_matrix(self, posts_content: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """Calculate similarity matrix for post content"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        post_ids = list(posts_content.keys())
        texts = list(posts_content.values())
        
        # Remove very short texts
        valid_posts = [(pid, text) for pid, text in zip(post_ids, texts) if len(text) > 10]
        
        if len(valid_posts) < 2:
            return {}
        
        valid_ids, valid_texts = zip(*valid_posts)
        
        # Calculate TF-IDF vectors
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            min_df=2,
            ngram_range=(1, 2)
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(valid_texts)
            similarity_matrix_np = cosine_similarity(tfidf_matrix)
            
            # Convert to dictionary format
            similarity_matrix = {}
            for i, post1 in enumerate(valid_ids):
                similarity_matrix[post1] = {}
                for j, post2 in enumerate(valid_ids):
                    if i != j:
                        similarity_matrix[post1][post2] = similarity_matrix_np[i, j]
            
            return similarity_matrix
            
        except Exception as e:
            logger.warning(f"Error calculating similarity matrix: {e}")
            return {}
    
    def _cluster_similar_content(self, similarity_matrix: Dict[str, Dict[str, float]], 
                               threshold: float, min_size: int) -> List[List[str]]:
        """Cluster content based on similarity threshold"""
        if not similarity_matrix:
            return []
        
        posts = list(similarity_matrix.keys())
        clusters = []
        assigned = set()
        
        for post in posts:
            if post in assigned:
                continue
            
            # Start new cluster
            cluster = [post]
            assigned.add(post)
            
            # Find similar posts
            for other_post in posts:
                if (other_post not in assigned and 
                    other_post in similarity_matrix[post] and
                    similarity_matrix[post][other_post] > threshold):
                    cluster.append(other_post)
                    assigned.add(other_post)
            
            if len(cluster) >= min_size:
                clusters.append(cluster)
        
        return clusters
    
    def _analyze_cluster_propagation(self, cluster: MisinformationCluster) -> None:
        """Analyze how content in the cluster propagated through the network"""
        
        # Get timestamps for cluster content
        timestamps = []
        post_times = {}
        
        for post_id in cluster.content_items:
            node = self.graph.get_node(post_id)
            if node and node.timestamp:
                timestamps.append(node.timestamp)
                post_times[post_id] = node.timestamp
        
        if len(timestamps) < 2:
            return
        
        timestamps.sort()
        time_span = (timestamps[-1] - timestamps[0]).total_seconds() / 3600  # hours
        
        # Calculate velocity (posts per hour)
        cluster.velocity = len(timestamps) / max(time_span, 1)
        
        # Find potential origin users (earliest posters)
        earliest_time = min(timestamps)
        origin_users = []
        
        for post_id, timestamp in post_times.items():
            if (timestamp - earliest_time).total_seconds() < 3600:  # Within 1 hour of earliest
                node = self.graph.get_node(post_id)
                if node:
                    author = node.get_attribute('author')
                    if author:
                        origin_users.append(author)
        
        cluster.origin_users = list(set(origin_users))
        
        # Analyze spread pattern
        cluster.spread_pattern = {
            'time_span_hours': time_span,
            'posts_per_hour': cluster.velocity,
            'origin_users_count': len(cluster.origin_users),
            'temporal_distribution': self._calculate_temporal_distribution(timestamps)
        }
    
    def _calculate_temporal_distribution(self, timestamps: List[datetime]) -> Dict[str, float]:
        """Calculate how content is distributed over time"""
        if len(timestamps) < 2:
            return {}
        
        timestamps.sort()
        intervals = []
        
        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i-1]).total_seconds() / 60  # minutes
            intervals.append(interval)
        
        return {
            'mean_interval_minutes': np.mean(intervals),
            'std_interval_minutes': np.std(intervals),
            'burst_ratio': len([i for i in intervals if i < 10]) / len(intervals)  # Posts within 10 minutes
        }
    
    def _is_cluster_suspicious(self, cluster: MisinformationCluster, velocity_threshold: float) -> bool:
        """Determine if a content cluster shows signs of coordination"""
        
        # Calculate coordination score based on multiple factors
        coordination_factors = []
        
        # Factor 1: High velocity (rapid spread)
        if cluster.velocity > velocity_threshold:
            coordination_factors.append(0.3)
        
        # Factor 2: High content similarity
        if cluster.similarity_score > 0.7:
            coordination_factors.append(0.25)
        
        # Factor 3: Few origin users relative to propagators
        if cluster.origin_users and len(cluster.origin_users) / len(cluster.propagators) < 0.3:
            coordination_factors.append(0.2)
        
        # Factor 4: Burst pattern (many posts in short time)
        if cluster.spread_pattern.get('burst_ratio', 0) > 0.5:
            coordination_factors.append(0.15)
        
        # Factor 5: Low variance in posting intervals (suggests automation)
        mean_interval = cluster.spread_pattern.get('mean_interval_minutes', float('inf'))
        std_interval = cluster.spread_pattern.get('std_interval_minutes', float('inf'))
        if mean_interval > 0 and std_interval / mean_interval < 0.5:
            coordination_factors.append(0.1)
        
        cluster.coordination_score = sum(coordination_factors)
        
        # Consider suspicious if coordination score > 0.5
        return cluster.coordination_score > 0.5
    
    def trace_propagation_paths(self, source_users: List[str], max_depth: int = 3) -> List[PropagationPath]:
        """Trace how information propagates from source users"""
        logger.info(f"Tracing propagation paths from {len(source_users)} source users...")
        
        propagation_paths = []
        
        for source_user in source_users:
            source_node = f"user_{source_user}"
            if source_node not in self.graph.nodes:
                continue
            
            # Use BFS to trace propagation
            visited = self.graph.bfs(source_node, max_depth)
            
            for target_node, depth in visited.items():
                if target_node.startswith('user_') and depth > 0:
                    target_user = target_node[5:]
                    
                    # Find shared content or interactions
                    shared_content = self._find_shared_content(source_user, target_user)
                    
                    for content_id, interaction_data in shared_content.items():
                        path = PropagationPath(
                            source_user=source_user,
                            target_user=target_user,
                            content_id=content_id,
                            timestamp=interaction_data.get('timestamp', datetime.now()),
                            platform='reddit',
                            interaction_type=interaction_data.get('type', 'unknown'),
                            delay_minutes=interaction_data.get('delay_minutes', 0),
                            influence_weight=1.0 / max(depth, 1)
                        )
                        propagation_paths.append(path)
        
        self.propagation_paths = propagation_paths
        logger.info(f"Traced {len(propagation_paths)} propagation paths")
        
        return propagation_paths
    
    def _find_shared_content(self, user1: str, user2: str) -> Dict[str, Dict[str, Any]]:
        """Find content shared between two users"""
        shared_content = {}
        
        # Get posts and comments from both users
        user1_content = self._get_user_content(user1)
        user2_content = self._get_user_content(user2)
        
        # Find temporal relationships
        for content1_id, content1_data in user1_content.items():
            for content2_id, content2_data in user2_content.items():
                
                # Check if they're related to the same post
                if (content1_data.get('post_id') == content2_data.get('post_id') or
                    content1_id == content2_data.get('post_id') or
                    content2_id == content1_data.get('post_id')):
                    
                    # Calculate time delay
                    if content1_data.get('timestamp') and content2_data.get('timestamp'):
                        delay = abs((content2_data['timestamp'] - content1_data['timestamp']).total_seconds() / 60)
                        
                        shared_content[f"{content1_id}_{content2_id}"] = {
                            'timestamp': min(content1_data['timestamp'], content2_data['timestamp']),
                            'delay_minutes': delay,
                            'type': 'post_interaction'
                        }
        
        return shared_content
    
    def _get_user_content(self, username: str) -> Dict[str, Dict[str, Any]]:
        """Get all content (posts/comments) from a user"""
        user_content = {}
        user_node = f"user_{username}"
        
        if user_node not in self.graph.nodes:
            return user_content
        
        # Get posts created by user
        for edge in self.graph.edges.get(user_node, []):
            if edge.edge_type == 'created' and edge.target.startswith('post_'):
                post_node = self.graph.get_node(edge.target)
                if post_node:
                    user_content[edge.target] = {
                        'type': 'post',
                        'timestamp': edge.timestamp,
                        'post_id': edge.target
                    }
        
        # Get comments by user
        for edge in self.graph.edges.get(user_node, []):
            if edge.edge_type == 'commented':
                user_content[f"comment_{edge.target}"] = {
                    'type': 'comment',
                    'timestamp': edge.timestamp,
                    'post_id': edge.target
                }
        
        return user_content
    
    def detect_bot_like_behavior(self, min_activity: int = 10) -> Dict[str, float]:
        """Detect users exhibiting bot-like behavior patterns"""
        logger.info("Detecting bot-like behavior patterns...")
        
        bot_scores = {}
        
        for node_id, node in self.graph.nodes.items():
            if node.node_type == 'user':
                username = node.get_attribute('username')
                if not username:
                    continue
                
                total_activity = node.get_attribute('total_activity', 0)
                
                if total_activity < min_activity:
                    continue
                
                # Calculate bot-like behavior score
                bot_score = self._calculate_bot_score(node)
                
                if bot_score > 0.5:  # Threshold for suspicious behavior
                    bot_scores[username] = bot_score
                    self.suspicious_users.add(username)
        
        logger.info(f"Detected {len(bot_scores)} users with bot-like behavior")
        return bot_scores
    
    def _calculate_bot_score(self, user_node) -> float:
        """Calculate bot-like behavior score for a user"""
        score_factors = []
        
        # Factor 1: High activity in short time span
        activity_span = user_node.get_attribute('activity_span_days', 1)
        total_activity = user_node.get_attribute('total_activity', 0)
        activity_rate = total_activity / max(activity_span, 1)
        
        if activity_rate > 50:  # More than 50 posts/comments per day
            score_factors.append(0.3)
        
        # Factor 2: Low content diversity
        subreddit_count = user_node.get_attribute('subreddit_count', 1)
        if subreddit_count < 3 and total_activity > 20:
            score_factors.append(0.2)
        
        # Factor 3: Very high or very low average scores
        avg_post_score = user_node.get_attribute('avg_post_score', 0)
        avg_comment_score = user_node.get_attribute('avg_comment_score', 0)
        
        if avg_post_score > 100 or avg_comment_score > 20:
            score_factors.append(0.2)
        
        # Factor 4: Suspicious username patterns
        username = user_node.get_attribute('username', '')
        if self._is_suspicious_username(username):
            score_factors.append(0.3)
        
        return min(sum(score_factors), 1.0)
    
    def _is_suspicious_username(self, username: str) -> bool:
        """Check if username follows suspicious patterns"""
        username_lower = username.lower()
        
        # Pattern 1: Random characters
        if re.match(r'^[a-z]{6,12}\d{1,4}$', username_lower):
            return True
        
        # Pattern 2: Word + numbers
        if re.match(r'^(word|user|account)\d+$', username_lower):
            return True
        
        # Pattern 3: Too many underscores or numbers
        if username.count('_') > 2 or sum(c.isdigit() for c in username) > len(username) * 0.5:
            return True
        
        return False
    
    def generate_misinformation_report(self) -> Dict[str, Any]:
        """Generate comprehensive misinformation analysis report"""
        logger.info("Generating misinformation analysis report...")
        
        report = {
            'summary': {
                'suspicious_clusters_detected': len(self.content_clusters),
                'propagation_paths_traced': len(self.propagation_paths),
                'suspicious_users_identified': len(self.suspicious_users),
                'analysis_timestamp': datetime.now().isoformat()
            },
            'content_clusters': [],
            'propagation_analysis': {},
            'suspicious_users': list(self.suspicious_users),
            'coordination_networks': self.coordination_networks,
            'recommendations': self._generate_misinformation_recommendations()
        }
        
        # Add cluster details
        for cluster in self.content_clusters:
            cluster_info = {
                'cluster_id': cluster.cluster_id,
                'content_items_count': len(cluster.content_items),
                'propagators_count': len(cluster.propagators),
                'origin_users': cluster.origin_users,
                'similarity_score': cluster.similarity_score,
                'velocity': cluster.velocity,
                'reach': cluster.reach,
                'coordination_score': cluster.coordination_score,
                'spread_pattern': cluster.spread_pattern
            }
            report['content_clusters'].append(cluster_info)
        
        # Propagation analysis
        if self.propagation_paths:
            report['propagation_analysis'] = {
                'total_paths': len(self.propagation_paths),
                'avg_delay_minutes': np.mean([p.delay_minutes for p in self.propagation_paths]),
                'most_influential_users': self._find_most_influential_users(),
                'fastest_propagation_time': min([p.delay_minutes for p in self.propagation_paths])
            }
        
        return report
    
    def _find_most_influential_users(self) -> List[Dict[str, Any]]:
        """Find users with highest influence in propagation"""
        influence_scores = defaultdict(float)
        
        for path in self.propagation_paths:
            influence_scores[path.source_user] += path.influence_weight
        
        # Sort by influence and return top 10
        sorted_users = sorted(influence_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {'username': user, 'influence_score': score}
            for user, score in sorted_users[:10]
        ]
    
    def _generate_misinformation_recommendations(self) -> List[str]:
        """Generate recommendations for misinformation mitigation"""
        recommendations = []
        
        if self.content_clusters:
            recommendations.append("Monitor content clusters with high similarity scores for coordinated messaging.")
            
            high_velocity_clusters = [c for c in self.content_clusters if c.velocity > 1.0]
            if high_velocity_clusters:
                recommendations.append("Investigate rapid-spreading content clusters for potential misinformation.")
        
        if self.suspicious_users:
            recommendations.append("Review accounts exhibiting bot-like behavior patterns.")
            recommendations.append("Implement rate limiting for high-activity accounts.")
        
        if self.propagation_paths:
            recommendations.append("Focus fact-checking efforts on high-influence propagation sources.")
        
        recommendations.extend([
            "Implement content verification systems for viral content.",
            "Promote authoritative sources in recommendation algorithms.",
            "Increase transparency in content moderation decisions."
        ])
        
        return recommendations


if __name__ == "__main__":
    # Example usage would go here
    pass
