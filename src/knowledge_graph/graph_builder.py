"""
Graph Builder Module for EchoGraph

This module converts Reddit data into a knowledge graph structure,
creating nodes for users, posts, subreddits, and URLs, and edges
for various types of interactions and relationships.
"""

from typing import Dict, List, Any, Set, Optional
from datetime import datetime
import json
import logging
from collections import defaultdict, Counter
import re

from knowledge_graph.graph import KnowledgeGraph, Node, Edge
from data_collection.reddit_scraper import RedditPost, RedditComment, RedditUser

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Builds knowledge graphs from social media data
    
    Features:
    - Convert Reddit data to graph structure
    - Create multi-type nodes (users, posts, subreddits, URLs, topics)
    - Generate weighted edges based on interactions
    - Extract implicit relationships
    - Handle temporal aspects
    """
    
    def __init__(self):
        self.graph = KnowledgeGraph()
        self.user_post_counts = defaultdict(int)
        self.user_comment_counts = defaultdict(int)
        self.subreddit_users = defaultdict(set)
        self.url_mentions = defaultdict(int)
        
    def build_from_reddit_data(self, data: Dict[str, Any]) -> KnowledgeGraph:
        """
        Build knowledge graph from Reddit data
        
        Args:
            data: Dictionary containing posts, comments, users, and interactions
            
        Returns:
            Populated KnowledgeGraph instance
        """
        logger.info("Building knowledge graph from Reddit data...")
        
        # Parse data if it's loaded from JSON
        posts = self._parse_posts(data.get('posts', []))
        comments = self._parse_comments(data.get('comments', []))
        users = data.get('users', [])
        
        # Create nodes
        self._create_user_nodes(posts, comments, users)
        self._create_subreddit_nodes(posts)
        self._create_post_nodes(posts)
        self._create_url_nodes(posts)
        self._create_topic_nodes(posts, comments)
        
        # Create edges
        self._create_user_post_edges(posts)
        self._create_user_comment_edges(comments)
        self._create_user_subreddit_edges(posts, comments)
        self._create_comment_reply_edges(comments)
        self._create_post_url_edges(posts)
        self._create_user_similarity_edges(posts, comments)
        self._create_content_similarity_edges(posts)
        
        logger.info(f"Knowledge graph built with {len(self.graph.nodes)} nodes and "
                   f"{sum(len(edges) for edges in self.graph.edges.values())} edges")
        
        return self.graph
    
    def _parse_posts(self, posts_data: List[Any]) -> List[RedditPost]:
        """Parse posts from JSON data or return as-is if already objects"""
        if not posts_data:
            return []
        
        if isinstance(posts_data[0], dict):
            # Convert from dict to RedditPost objects
            posts = []
            for post_dict in posts_data:
                post = RedditPost(
                    id=post_dict['id'],
                    title=post_dict['title'],
                    author=post_dict['author'],
                    subreddit=post_dict['subreddit'],
                    score=post_dict['score'],
                    num_comments=post_dict['num_comments'],
                    created_utc=datetime.fromisoformat(post_dict['created_utc']),
                    selftext=post_dict.get('selftext', ''),
                    url=post_dict.get('url', ''),
                    is_self=post_dict.get('is_self', True),
                    upvote_ratio=post_dict.get('upvote_ratio', 0.5),
                    flair=post_dict.get('flair'),
                    gilded=post_dict.get('gilded', 0)
                )
                posts.append(post)
            return posts
        else:
            return posts_data
    
    def _parse_comments(self, comments_data: List[Any]) -> List[RedditComment]:
        """Parse comments from JSON data or return as-is if already objects"""
        if not comments_data:
            return []
        
        if isinstance(comments_data[0], dict):
            # Convert from dict to RedditComment objects
            comments = []
            for comment_dict in comments_data:
                comment = RedditComment(
                    id=comment_dict['id'],
                    body=comment_dict['body'],
                    author=comment_dict['author'],
                    post_id=comment_dict['post_id'],
                    parent_id=comment_dict['parent_id'],
                    score=comment_dict['score'],
                    created_utc=datetime.fromisoformat(comment_dict['created_utc']),
                    is_submitter=comment_dict.get('is_submitter', False),
                    gilded=comment_dict.get('gilded', 0),
                    depth=comment_dict.get('depth', 0)
                )
                comments.append(comment)
            return comments
        else:
            return comments_data
    
    def _create_user_nodes(self, posts: List[RedditPost], comments: List[RedditComment], 
                          users: List[Any]) -> None:
        """Create user nodes with aggregated statistics"""
        logger.info("Creating user nodes...")
        
        # Collect user statistics
        user_stats = defaultdict(lambda: {
            'posts': 0, 'comments': 0, 'total_post_score': 0, 'total_comment_score': 0,
            'subreddits': set(), 'first_seen': None, 'last_seen': None, 'gilded_posts': 0,
            'gilded_comments': 0
        })
        
        # Process posts
        for post in posts:
            if post.author and post.author != '[deleted]':
                stats = user_stats[post.author]
                stats['posts'] += 1
                stats['total_post_score'] += post.score
                stats['subreddits'].add(post.subreddit)
                stats['gilded_posts'] += post.gilded
                
                if stats['first_seen'] is None or post.created_utc < stats['first_seen']:
                    stats['first_seen'] = post.created_utc
                if stats['last_seen'] is None or post.created_utc > stats['last_seen']:
                    stats['last_seen'] = post.created_utc
        
        # Process comments
        for comment in comments:
            if comment.author and comment.author != '[deleted]':
                stats = user_stats[comment.author]
                stats['comments'] += 1
                stats['total_comment_score'] += comment.score
                stats['gilded_comments'] += comment.gilded
                
                if stats['first_seen'] is None or comment.created_utc < stats['first_seen']:
                    stats['first_seen'] = comment.created_utc
                if stats['last_seen'] is None or comment.created_utc > stats['last_seen']:
                    stats['last_seen'] = comment.created_utc
        
        # Create user nodes
        for username, stats in user_stats.items():
            if username != '[deleted]':
                attributes = {
                    'username': username,
                    'post_count': stats['posts'],
                    'comment_count': stats['comments'],
                    'total_posts_score': stats['total_post_score'],
                    'total_comments_score': stats['total_comment_score'],
                    'avg_post_score': stats['total_post_score'] / max(stats['posts'], 1),
                    'avg_comment_score': stats['total_comment_score'] / max(stats['comments'], 1),
                    'subreddit_count': len(stats['subreddits']),
                    'subreddits': list(stats['subreddits']),
                    'activity_span_days': (stats['last_seen'] - stats['first_seen']).days if stats['first_seen'] and stats['last_seen'] else 0,
                    'total_activity': stats['posts'] + stats['comments'],
                    'gilded_content': stats['gilded_posts'] + stats['gilded_comments']
                }
                
                node = Node(
                    id=f"user_{username}",
                    node_type="user",
                    attributes=attributes,
                    timestamp=stats['first_seen']
                )
                self.graph.add_node(node)
        
        logger.info(f"Created {len(user_stats)} user nodes")
    
    def _create_subreddit_nodes(self, posts: List[RedditPost]) -> None:
        """Create subreddit nodes with statistics"""
        logger.info("Creating subreddit nodes...")
        
        subreddit_stats = defaultdict(lambda: {
            'post_count': 0, 'total_score': 0, 'total_comments': 0,
            'users': set(), 'topics': set()
        })
        
        for post in posts:
            stats = subreddit_stats[post.subreddit]
            stats['post_count'] += 1
            stats['total_score'] += post.score
            stats['total_comments'] += post.num_comments
            stats['users'].add(post.author)
            
            # Extract topics from titles and flairs
            if post.flair:
                stats['topics'].add(post.flair.lower())
            
            # Simple topic extraction from titles
            title_words = set(re.findall(r'\b\w+\b', post.title.lower()))
            common_topics = title_words.intersection({
                'trump', 'biden', 'election', 'covid', 'vaccine', 'climate', 'economy',
                'immigration', 'healthcare', 'education', 'gun', 'abortion', 'tax'
            })
            stats['topics'].update(common_topics)
        
        for subreddit, stats in subreddit_stats.items():
            attributes = {
                'name': subreddit,
                'post_count': stats['post_count'],
                'total_score': stats['total_score'],
                'avg_score': stats['total_score'] / max(stats['post_count'], 1),
                'total_comments': stats['total_comments'],
                'avg_comments': stats['total_comments'] / max(stats['post_count'], 1),
                'unique_users': len(stats['users']),
                'activity_density': stats['post_count'] / max(len(stats['users']), 1),
                'topics': list(stats['topics'])
            }
            
            node = Node(
                id=f"subreddit_{subreddit}",
                node_type="subreddit",
                attributes=attributes
            )
            self.graph.add_node(node)
        
        logger.info(f"Created {len(subreddit_stats)} subreddit nodes")
    
    def _create_post_nodes(self, posts: List[RedditPost]) -> None:
        """Create post nodes"""
        logger.info("Creating post nodes...")
        
        for post in posts:
            attributes = {
                'title': post.title,
                'author': post.author,
                'subreddit': post.subreddit,
                'score': post.score,
                'num_comments': post.num_comments,
                'upvote_ratio': post.upvote_ratio,
                'is_self': post.is_self,
                'flair': post.flair,
                'gilded': post.gilded,
                'has_external_url': not post.is_self and post.url,
                'text_length': len(post.selftext) if post.selftext else 0,
                'title_length': len(post.title),
                'created_utc': post.created_utc.isoformat()
            }
            
            node = Node(
                id=f"post_{post.id}",
                node_type="post",
                attributes=attributes,
                timestamp=post.created_utc
            )
            self.graph.add_node(node)
        
        logger.info(f"Created {len(posts)} post nodes")
    
    def _create_url_nodes(self, posts: List[RedditPost]) -> None:
        """Create URL nodes for external links"""
        logger.info("Creating URL nodes...")
        
        url_stats = defaultdict(lambda: {
            'mentions': 0, 'domains': set(), 'posts': [], 'users': set(),
            'subreddits': set(), 'total_score': 0
        })
        
        for post in posts:
            if not post.is_self and post.url:
                # Extract domain
                domain = post.url.split('/')[2] if '/' in post.url else post.url
                
                stats = url_stats[post.url]
                stats['mentions'] += 1
                stats['domains'].add(domain)
                stats['posts'].append(post.id)
                stats['users'].add(post.author)
                stats['subreddits'].add(post.subreddit)
                stats['total_score'] += post.score
        
        for url, stats in url_stats.items():
            if stats['mentions'] > 0:  # Only create nodes for mentioned URLs
                domain = list(stats['domains'])[0] if stats['domains'] else 'unknown'
                
                attributes = {
                    'url': url,
                    'domain': domain,
                    'mention_count': stats['mentions'],
                    'unique_users': len(stats['users']),
                    'unique_subreddits': len(stats['subreddits']),
                    'total_score': stats['total_score'],
                    'avg_score': stats['total_score'] / stats['mentions'],
                    'type': self._categorize_domain(domain)
                }
                
                node = Node(
                    id=f"url_{hash(url) % 1000000}",  # Simple hash for ID
                    node_type="url",
                    attributes=attributes
                )
                self.graph.add_node(node)
        
        logger.info(f"Created {len(url_stats)} URL nodes")
    
    def _create_topic_nodes(self, posts: List[RedditPost], comments: List[RedditComment]) -> None:
        """Create topic nodes based on content analysis"""
        logger.info("Creating topic nodes...")
        
        # Political and controversial topics to track
        topics = {
            'election': ['election', 'vote', 'voting', 'ballot', 'candidate', 'campaign'],
            'covid': ['covid', 'coronavirus', 'pandemic', 'vaccine', 'mask', 'lockdown'],
            'climate': ['climate', 'global warming', 'environment', 'carbon', 'greenhouse'],
            'economy': ['economy', 'recession', 'inflation', 'unemployment', 'market', 'stock'],
            'immigration': ['immigration', 'border', 'immigrant', 'refugee', 'asylum'],
            'healthcare': ['healthcare', 'insurance', 'medical', 'hospital', 'doctor'],
            'gun': ['gun', 'firearm', 'shooting', 'rifle', 'pistol', 'ammunition'],
            'abortion': ['abortion', 'reproductive', 'pregnancy', 'fetus', 'pro-life', 'pro-choice'],
            'tax': ['tax', 'taxation', 'revenue', 'irs', 'income tax', 'corporate tax']
        }
        
        topic_stats = defaultdict(lambda: {
            'mentions': 0, 'posts': set(), 'users': set(), 'subreddits': set(),
            'total_score': 0, 'sentiment_scores': []
        })
        
        # Analyze posts
        for post in posts:
            text = (post.title + ' ' + post.selftext).lower()
            for topic, keywords in topics.items():
                if any(keyword in text for keyword in keywords):
                    stats = topic_stats[topic]
                    stats['mentions'] += 1
                    stats['posts'].add(post.id)
                    stats['users'].add(post.author)
                    stats['subreddits'].add(post.subreddit)
                    stats['total_score'] += post.score
        
        # Analyze comments
        for comment in comments:
            text = comment.body.lower()
            for topic, keywords in topics.items():
                if any(keyword in text for keyword in keywords):
                    stats = topic_stats[topic]
                    stats['mentions'] += 1
                    stats['users'].add(comment.author)
        
        # Create topic nodes
        for topic, stats in topic_stats.items():
            if stats['mentions'] > 0:
                attributes = {
                    'topic': topic,
                    'mention_count': stats['mentions'],
                    'unique_posts': len(stats['posts']),
                    'unique_users': len(stats['users']),
                    'unique_subreddits': len(stats['subreddits']),
                    'total_score': stats['total_score'],
                    'avg_score': stats['total_score'] / max(len(stats['posts']), 1),
                    'controversy_level': self._calculate_controversy(stats)
                }
                
                node = Node(
                    id=f"topic_{topic}",
                    node_type="topic",
                    attributes=attributes
                )
                self.graph.add_node(node)
        
        logger.info(f"Created {len(topic_stats)} topic nodes")
    
    def _calculate_controversy(self, stats: Dict[str, Any]) -> float:
        """Calculate controversy level for a topic"""
        # Simple heuristic: more subreddits + lower average score = higher controversy
        subreddit_diversity = len(stats['subreddits'])
        avg_score = stats['total_score'] / max(len(stats['posts']), 1)
        
        # Normalize and combine metrics
        controversy = (subreddit_diversity / 10) + max(0, (10 - avg_score) / 10)
        return min(controversy, 1.0)
    
    def _categorize_domain(self, domain: str) -> str:
        """Categorize domain type"""
        if any(news in domain for news in ['cnn', 'fox', 'bbc', 'reuters', 'nytimes']):
            return 'mainstream_news'
        elif any(social in domain for social in ['twitter', 'facebook', 'youtube']):
            return 'social_media'
        elif 'wiki' in domain:
            return 'wiki'
        elif any(ext in domain for ext in ['.gov', '.edu', '.org']):
            return 'institutional'
        else:
            return 'other'
    
    def _create_user_post_edges(self, posts: List[RedditPost]) -> None:
        """Create edges between users and their posts"""
        for post in posts:
            if post.author != '[deleted]':
                edge = Edge(
                    source=f"user_{post.author}",
                    target=f"post_{post.id}",
                    edge_type="created",
                    weight=1.0,
                    timestamp=post.created_utc
                )
                edge.add_attribute('score', post.score)
                edge.add_attribute('num_comments', post.num_comments)
                self.graph.add_edge(edge)
    
    def _create_user_comment_edges(self, comments: List[RedditComment]) -> None:
        """Create edges for user commenting behavior"""
        for comment in comments:
            if comment.author != '[deleted]':
                # Edge from user to post (commented on)
                edge = Edge(
                    source=f"user_{comment.author}",
                    target=f"post_{comment.post_id}",
                    edge_type="commented",
                    weight=1.0,
                    timestamp=comment.created_utc
                )
                edge.add_attribute('score', comment.score)
                edge.add_attribute('is_submitter', comment.is_submitter)
                self.graph.add_edge(edge)
    
    def _create_user_subreddit_edges(self, posts: List[RedditPost], 
                                   comments: List[RedditComment]) -> None:
        """Create edges between users and subreddits"""
        user_subreddit_activity = defaultdict(lambda: defaultdict(int))
        
        # Count activities per subreddit
        for post in posts:
            if post.author != '[deleted]':
                user_subreddit_activity[post.author][post.subreddit] += 2  # Posts worth more
        
        for comment in comments:
            if comment.author != '[deleted]':
                # Find post to get subreddit
                post_subreddit = None
                for post in posts:
                    if post.id == comment.post_id:
                        post_subreddit = post.subreddit
                        break
                
                if post_subreddit:
                    user_subreddit_activity[comment.author][post_subreddit] += 1
        
        # Create edges
        for user, subreddit_counts in user_subreddit_activity.items():
            for subreddit, activity_count in subreddit_counts.items():
                edge = Edge(
                    source=f"user_{user}",
                    target=f"subreddit_{subreddit}",
                    edge_type="active_in",
                    weight=min(activity_count / 10.0, 1.0)  # Normalize weight
                )
                edge.add_attribute('activity_count', activity_count)
                self.graph.add_edge(edge)
    
    def _create_comment_reply_edges(self, comments: List[RedditComment]) -> None:
        """Create edges for comment reply relationships"""
        comment_lookup = {comment.id: comment for comment in comments}
        
        for comment in comments:
            if comment.parent_id.startswith('t1_'):  # Reply to another comment
                parent_comment_id = comment.parent_id[3:]  # Remove 't1_' prefix
                if parent_comment_id in comment_lookup:
                    parent_comment = comment_lookup[parent_comment_id]
                    
                    if (comment.author != '[deleted]' and 
                        parent_comment.author != '[deleted]' and
                        comment.author != parent_comment.author):
                        
                        edge = Edge(
                            source=f"user_{comment.author}",
                            target=f"user_{parent_comment.author}",
                            edge_type="replied_to",
                            weight=1.0,
                            timestamp=comment.created_utc
                        )
                        edge.add_attribute('post_id', comment.post_id)
                        self.graph.add_edge(edge)
    
    def _create_post_url_edges(self, posts: List[RedditPost]) -> None:
        """Create edges between posts and URLs"""
        for post in posts:
            if not post.is_self and post.url:
                url_id = f"url_{hash(post.url) % 1000000}"
                
                edge = Edge(
                    source=f"post_{post.id}",
                    target=url_id,
                    edge_type="links_to",
                    weight=1.0,
                    timestamp=post.created_utc
                )
                edge.add_attribute('score', post.score)
                self.graph.add_edge(edge)
    
    def _create_user_similarity_edges(self, posts: List[RedditPost], 
                                    comments: List[RedditComment]) -> None:
        """Create similarity edges between users based on behavior"""
        logger.info("Creating user similarity edges...")
        
        # Calculate user activity patterns
        user_subreddit_activity = defaultdict(set)
        user_topic_activity = defaultdict(set)
        
        for post in posts:
            if post.author != '[deleted]':
                user_subreddit_activity[post.author].add(post.subreddit)
                # Simple topic extraction
                title_lower = post.title.lower()
                if any(word in title_lower for word in ['trump', 'biden', 'election']):
                    user_topic_activity[post.author].add('politics')
                if any(word in title_lower for word in ['covid', 'vaccine', 'mask']):
                    user_topic_activity[post.author].add('health')
        
        # Create similarity edges based on shared subreddits and topics
        users = list(user_subreddit_activity.keys())
        for i, user1 in enumerate(users):
            for user2 in users[i+1:]:
                if user1 != user2:
                    # Calculate Jaccard similarity for subreddits
                    subreddits1 = user_subreddit_activity[user1]
                    subreddits2 = user_subreddit_activity[user2]
                    
                    if subreddits1 and subreddits2:
                        subreddit_similarity = len(subreddits1.intersection(subreddits2)) / \
                                             len(subreddits1.union(subreddits2))
                        
                        # Calculate topic similarity
                        topics1 = user_topic_activity[user1]
                        topics2 = user_topic_activity[user2]
                        topic_similarity = 0.0
                        
                        if topics1 and topics2:
                            topic_similarity = len(topics1.intersection(topics2)) / \
                                             len(topics1.union(topics2))
                        
                        # Combined similarity
                        total_similarity = (subreddit_similarity + topic_similarity) / 2
                        
                        if total_similarity > 0.3:  # Threshold for creating edge
                            edge = Edge(
                                source=f"user_{user1}",
                                target=f"user_{user2}",
                                edge_type="similar_to",
                                weight=total_similarity
                            )
                            edge.add_attribute('subreddit_similarity', subreddit_similarity)
                            edge.add_attribute('topic_similarity', topic_similarity)
                            self.graph.add_edge(edge)
    
    def _create_content_similarity_edges(self, posts: List[RedditPost]) -> None:
        """Create edges between posts with similar content"""
        logger.info("Creating content similarity edges...")
        
        # Simple content similarity based on title word overlap
        post_words = {}
        for post in posts:
            words = set(re.findall(r'\b\w+\b', post.title.lower()))
            # Remove common words
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
            words = words - common_words
            post_words[post.id] = words
        
        # Calculate similarities
        posts_list = list(posts)
        for i, post1 in enumerate(posts_list):
            for post2 in posts_list[i+1:]:
                if post1.id != post2.id:
                    words1 = post_words.get(post1.id, set())
                    words2 = post_words.get(post2.id, set())
                    
                    if words1 and words2:
                        similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                        
                        if similarity > 0.4:  # Threshold for similar content
                            edge = Edge(
                                source=f"post_{post1.id}",
                                target=f"post_{post2.id}",
                                edge_type="similar_content",
                                weight=similarity
                            )
                            edge.add_attribute('word_overlap', len(words1.intersection(words2)))
                            self.graph.add_edge(edge)


def load_reddit_data(filepath: str) -> Dict[str, Any]:
    """Load Reddit data from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":
    # Example usage
    builder = GraphBuilder()
    
    # This would normally load real data
    # data = load_reddit_data('reddit_data.json')
    
    # For demo, create sample data
    from ..data_collection.reddit_scraper import RedditScraper
    scraper = RedditScraper()
    data = scraper.generate_demo_data(['politics', 'conspiracy'], 50)
    
    # Build graph
    kg = builder.build_from_reddit_data(data)
    
    print("Graph Statistics:", kg.get_graph_statistics())
    
    # Save graph
    kg.save_to_file('knowledge_graph.pkl')
