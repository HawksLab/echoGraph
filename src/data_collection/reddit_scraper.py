"""
Reddit Data Collection Module for EchoGraph

This module handles scraping Reddit data for social network analysis.
It collects posts, comments, user interactions, and builds the initial
data structure for knowledge graph construction.
"""

import praw
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
import time
import json
import requests
from dataclasses import dataclass
import re
from urllib.parse import urlparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RedditPost:
    """Structure for Reddit post data"""
    id: str
    title: str
    author: str
    subreddit: str
    score: int
    num_comments: int
    created_utc: datetime
    selftext: str
    url: str
    is_self: bool
    upvote_ratio: float
    flair: Optional[str] = None
    gilded: int = 0
    awards: List[Dict] = None


@dataclass
class RedditComment:
    """Structure for Reddit comment data"""
    id: str
    body: str
    author: str
    post_id: str
    parent_id: str
    score: int
    created_utc: datetime
    is_submitter: bool
    gilded: int = 0
    depth: int = 0


@dataclass
class RedditUser:
    """Structure for Reddit user data"""
    username: str
    comment_karma: int
    link_karma: int
    created_utc: datetime
    is_verified: bool
    is_mod: bool
    has_verified_email: bool
    subreddits: List[str] = None


class RedditScraper:
    """
    Reddit data scraper for social network analysis
    
    Features:
    - Scrape posts and comments from specified subreddits
    - Collect user interaction data
    - Extract URLs and external links
    - Build temporal datasets
    - Handle rate limiting
    """
    
    def __init__(self, client_id: str = None, client_secret: str = None, user_agent: str = None):
        """
        Initialize Reddit scraper
        
        Args:
            client_id: Reddit API client ID (optional)
            client_secret: Reddit API client secret (optional)
            user_agent: User agent string
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent or "EchoGraph Social Analysis Tool v1.0"
        
        # Try to initialize PRAW if credentials are provided
        self.reddit = None
        self.use_api = False
        
        if client_id and client_secret:
            try:
                self.reddit = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent=user_agent
                )
                self.use_api = True
                logger.info("Reddit API initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Reddit API: {e}")
        
        # Use public Reddit JSON API as default
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})
        logger.info("Initialized Reddit scraper with public JSON API")
    
    def collect_subreddit_data_public(self, subreddits: List[str], limit: int = 100) -> Dict[str, Any]:
        """
        Collect Reddit data using public JSON API (no authentication required)
        
        Args:
            subreddits: List of subreddit names
            limit: Number of posts to collect per subreddit
            
        Returns:
            Dictionary containing posts, comments, and user data
        """
        all_posts = []
        all_comments = []
        all_users = set()
        user_interactions = []
        
        for subreddit_name in subreddits:
            logger.info(f"Collecting data from r/{subreddit_name} using public API")
            
            try:
                # Get posts from subreddit
                url = f"https://www.reddit.com/r/{subreddit_name}/hot.json?limit={min(limit, 100)}"
                response = self.session.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    posts = data.get('data', {}).get('children', [])
                    
                    for post_data in posts:
                        post = post_data['data']
                        
                        # Create RedditPost object
                        reddit_post = RedditPost(
                            id=post['id'],
                            title=post['title'],
                            author=post.get('author', '[deleted]'),
                            subreddit=post['subreddit'],
                            score=post['score'],
                            num_comments=post['num_comments'],
                            created_utc=datetime.fromtimestamp(post['created_utc']),
                            selftext=post.get('selftext', ''),
                            url=post['url'],
                            is_self=post['is_self'],
                            upvote_ratio=post.get('upvote_ratio', 0.5),
                            flair=post.get('link_flair_text'),
                            gilded=post.get('gilded', 0)
                        )
                        
                        all_posts.append(reddit_post)
                        all_users.add(reddit_post.author)
                        
                        # Get comments for this post
                        if post['num_comments'] > 0:
                            comments = self._get_post_comments_public(post['id'])
                            all_comments.extend(comments)
                            
                            # Track interactions
                            for comment in comments:
                                all_users.add(comment.author)
                                user_interactions.append({
                                    'source': comment.author,
                                    'target': reddit_post.author,
                                    'type': 'replied_to',
                                    'post_id': reddit_post.id,
                                    'timestamp': comment.created_utc
                                })
                        
                        # Add small delay to be respectful
                        time.sleep(0.1)
                        
                else:
                    logger.warning(f"Failed to fetch data from r/{subreddit_name}: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Error collecting from r/{subreddit_name}: {e}")
                continue
            
            # Rate limiting
            time.sleep(1)
        
        logger.info(f"Collected {len(all_posts)} posts and {len(all_comments)} comments from {len(all_users)} users")
        
        return {
            'posts': all_posts,
            'comments': all_comments,
            'users': list(all_users),
            'user_interactions': user_interactions,
            'metadata': {
                'collection_time': datetime.now(),
                'subreddits': subreddits,
                'data_source': 'reddit_public_api'
            }
        }
    
    def _get_post_comments_public(self, post_id: str, limit: int = 50) -> List[RedditComment]:
        """Get comments for a specific post using public API"""
        comments = []
        
        try:
            url = f"https://www.reddit.com/comments/{post_id}.json?limit={limit}"
            response = self.session.get(url)
            
            if response.status_code == 200:
                data = response.json()
                if len(data) > 1:  # Check if comments exist
                    comment_data = data[1]['data']['children']
                    
                    for comment_item in comment_data:
                        if comment_item['kind'] == 't1':  # Comment type
                            comment = comment_item['data']
                            
                            if comment.get('author') != '[deleted]' and comment.get('body'):
                                reddit_comment = RedditComment(
                                    id=comment['id'],
                                    body=comment['body'],
                                    author=comment.get('author', '[deleted]'),
                                    post_id=post_id,
                                    parent_id=comment.get('parent_id', ''),
                                    score=comment.get('score', 0),
                                    created_utc=datetime.fromtimestamp(comment['created_utc']),
                                    is_submitter=comment.get('is_submitter', False),
                                    depth=comment.get('depth', 0)
                                )
                                comments.append(reddit_comment)
                                
                                if len(comments) >= limit:
                                    break
                                    
        except Exception as e:
            logger.warning(f"Error getting comments for post {post_id}: {e}")
        
        return comments
    
    def generate_demo_data(self, subreddits: List[str], num_posts: int = 100) -> Dict[str, Any]:
        """
        Generate realistic demo data for testing when Reddit API is not available
        """
        logger.info("Generating demo Reddit data...")
        
        import random
        from faker import Faker
        fake = Faker()
        
        # Sample political/controversial topics
        topics = [
            "climate change policy", "election results", "healthcare reform",
            "immigration policy", "economic inequality", "social media regulation",
            "cryptocurrency adoption", "AI ethics", "privacy rights", "education funding"
        ]
        
        # Sample usernames with different political leanings
        usernames = [
            "progressive_voter", "conservative_voice", "moderate_thinker", "independent_mind",
            "liberal_activist", "traditional_values", "young_democrat", "fiscal_conservative",
            "green_party_member", "libertarian_view", "social_democrat", "tea_party_patriot"
        ]
        
        posts = []
        comments = []
        users = []
        user_interactions = []
        
        # Generate users
        for username in usernames:
            user = RedditUser(
                username=username,
                comment_karma=random.randint(100, 10000),
                link_karma=random.randint(50, 5000),
                created_utc=fake.date_time_between(start_date='-2y', end_date='now'),
                is_verified=random.choice([True, False]),
                is_mod=random.choice([True, False]) if random.random() < 0.1 else False,
                has_verified_email=random.choice([True, False]),
                subreddits=random.sample(subreddits, k=random.randint(1, len(subreddits)))
            )
            users.append(user)
        
        # Generate posts
        for i in range(num_posts):
            subreddit = random.choice(subreddits)
            author = random.choice(usernames)
            topic = random.choice(topics)
            
            # Create realistic titles based on subreddit and topic
            if subreddit == "politics":
                title_templates = [
                    f"New study shows impact of {topic} on communities",
                    f"Breaking: Major development in {topic} legislation",
                    f"Opinion: Why {topic} matters for our future",
                    f"Analysis: The real truth about {topic}",
                ]
            elif subreddit == "conspiracy":
                title_templates = [
                    f"Hidden agenda behind {topic} exposed",
                    f"What they don't want you to know about {topic}",
                    f"The real story behind {topic} coverup",
                    f"Exclusive: Inside sources reveal {topic} truth",
                ]
            else:
                title_templates = [
                    f"Discussion: Thoughts on {topic}?",
                    f"Question about {topic}",
                    f"My experience with {topic}",
                    f"Resources for understanding {topic}",
                ]
            
            title = random.choice(title_templates)
            
            post = RedditPost(
                id=f"post_{i}",
                title=title,
                author=author,
                subreddit=subreddit,
                score=random.randint(-50, 1000),
                num_comments=random.randint(0, 200),
                created_utc=fake.date_time_between(start_date='-30d', end_date='now'),
                selftext=fake.text(max_nb_chars=500) if random.random() < 0.7 else "",
                url=fake.url() if random.random() < 0.3 else "",
                is_self=random.choice([True, False]),
                upvote_ratio=random.uniform(0.5, 0.95),
                flair=random.choice([None, "Discussion", "News", "Opinion", "Question"]),
                gilded=random.randint(0, 3) if random.random() < 0.1 else 0
            )
            posts.append(post)
            
            # Generate comments for this post
            num_comments = min(post.num_comments, random.randint(5, 30))
            for j in range(num_comments):
                comment_author = random.choice(usernames)
                
                comment = RedditComment(
                    id=f"comment_{i}_{j}",
                    body=fake.text(max_nb_chars=200),
                    author=comment_author,
                    post_id=post.id,
                    parent_id=post.id if j == 0 else f"comment_{i}_{random.randint(0, j-1)}",
                    score=random.randint(-20, 100),
                    created_utc=post.created_utc + timedelta(minutes=random.randint(1, 1440)),
                    is_submitter=(comment_author == post.author),
                    gilded=random.randint(0, 1) if random.random() < 0.05 else 0,
                    depth=random.randint(0, 3)
                )
                comments.append(comment)
        
        # Generate user interactions (follows, upvotes, etc.)
        for _ in range(num_posts * 2):
            user1 = random.choice(usernames)
            user2 = random.choice(usernames)
            if user1 != user2:
                interaction_type = random.choice(['upvote', 'downvote', 'reply', 'mention'])
                user_interactions.append({
                    'source_user': user1,
                    'target_user': user2,
                    'interaction_type': interaction_type,
                    'timestamp': fake.date_time_between(start_date='-30d', end_date='now'),
                    'post_id': random.choice([p.id for p in posts]) if random.random() < 0.8 else None
                })
        
        logger.info(f"Generated {len(posts)} posts, {len(comments)} comments, {len(users)} users")
        
        return {
            'posts': posts,
            'comments': comments,
            'users': users,
            'user_interactions': user_interactions,
            'subreddits': subreddits,
            'collection_timestamp': datetime.now()
        }
    
    def collect_subreddit_data(self, subreddits: List[str], time_filter: str = 'month', 
                             limit: int = 100) -> Dict[str, Any]:
        """
        Collect posts and comments from specified subreddits
        
        Args:
            subreddits: List of subreddit names to scrape
            time_filter: Time filter ('hour', 'day', 'week', 'month', 'year', 'all')
            limit: Maximum number of posts per subreddit
            
        Returns:
            Dictionary containing posts, comments, and user data
        """
        if self.use_api and self.reddit:
            return self._collect_with_api(subreddits, time_filter, limit)
        else:
            # Use public JSON API as primary method
            logger.info("Using Reddit public JSON API")
            return self.collect_subreddit_data_public(subreddits, limit)
    
    def _collect_with_api(self, subreddits: List[str], time_filter: str, limit: int) -> Dict[str, Any]:
        """Collect data using Reddit API"""
        all_posts = []
        all_comments = []
        all_users = set()
        user_interactions = []
        
        for subreddit_name in subreddits:
            logger.info(f"Collecting data from r/{subreddit_name}")
            
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Get hot posts
                for submission in subreddit.hot(limit=limit):
                    # Convert submission to our format
                    post = RedditPost(
                        id=submission.id,
                        title=submission.title,
                        author=str(submission.author) if submission.author else '[deleted]',
                        subreddit=subreddit_name,
                        score=submission.score,
                        num_comments=submission.num_comments,
                        created_utc=datetime.fromtimestamp(submission.created_utc),
                        selftext=submission.selftext,
                        url=submission.url,
                        is_self=submission.is_self,
                        upvote_ratio=submission.upvote_ratio,
                        flair=submission.link_flair_text,
                        gilded=submission.gilded
                    )
                    all_posts.append(post)
                    all_users.add(post.author)
                    
                    # Get comments
                    try:
                        submission.comments.replace_more(limit=0)
                        for comment in submission.comments.list()[:50]:  # Limit comments
                            if comment.author:
                                reddit_comment = RedditComment(
                                    id=comment.id,
                                    body=comment.body,
                                    author=str(comment.author),
                                    post_id=submission.id,
                                    parent_id=comment.parent_id,
                                    score=comment.score,
                                    created_utc=datetime.fromtimestamp(comment.created_utc),
                                    is_submitter=(str(comment.author) == str(submission.author)),
                                    gilded=comment.gilded
                                )
                                all_comments.append(reddit_comment)
                                all_users.add(reddit_comment.author)
                    except Exception as e:
                        logger.warning(f"Error collecting comments for {submission.id}: {e}")
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error collecting from r/{subreddit_name}: {e}")
        
        logger.info(f"Collected {len(all_posts)} posts and {len(all_comments)} comments")
        
        return {
            'posts': all_posts,
            'comments': all_comments,
            'users': list(all_users),
            'user_interactions': user_interactions,
            'subreddits': subreddits,
            'collection_timestamp': datetime.now()
        }
    
    def extract_urls_from_posts(self, posts: List[RedditPost]) -> List[Dict[str, Any]]:
        """Extract and categorize URLs from posts"""
        url_data = []
        
        for post in posts:
            urls = []
            
            # Extract from URL field
            if post.url and not post.is_self:
                urls.append(post.url)
            
            # Extract from selftext
            if post.selftext:
                url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
                urls.extend(re.findall(url_pattern, post.selftext))
            
            for url in urls:
                try:
                    parsed = urlparse(url)
                    domain = parsed.netloc.lower()
                    
                    # Categorize URL type
                    url_type = self._categorize_url(domain)
                    
                    url_data.append({
                        'url': url,
                        'domain': domain,
                        'type': url_type,
                        'post_id': post.id,
                        'author': post.author,
                        'subreddit': post.subreddit,
                        'timestamp': post.created_utc
                    })
                except Exception as e:
                    logger.warning(f"Error parsing URL {url}: {e}")
        
        return url_data
    
    def _categorize_url(self, domain: str) -> str:
        """Categorize URL by domain"""
        news_domains = [
            'cnn.com', 'foxnews.com', 'bbc.com', 'reuters.com', 'ap.org',
            'nytimes.com', 'washingtonpost.com', 'wsj.com', 'npr.org'
        ]
        
        social_domains = [
            'twitter.com', 'facebook.com', 'instagram.com', 'tiktok.com',
            'youtube.com', 'reddit.com'
        ]
        
        blog_domains = [
            'medium.com', 'substack.com', 'wordpress.com', 'blogspot.com'
        ]
        
        if any(news_domain in domain for news_domain in news_domains):
            return 'mainstream_news'
        elif any(social_domain in domain for social_domain in social_domains):
            return 'social_media'
        elif any(blog_domain in domain for blog_domain in blog_domains):
            return 'blog'
        elif 'wiki' in domain:
            return 'wiki'
        elif any(ext in domain for ext in ['.gov', '.edu', '.org']):
            return 'institutional'
        else:
            return 'other'
    
    def save_data(self, data: Dict[str, Any], filepath: str) -> None:
        """Save collected data to file"""
        # Convert dataclasses to dictionaries for JSON serialization
        serializable_data = {
            'posts': [self._post_to_dict(post) for post in data['posts']],
            'comments': [self._comment_to_dict(comment) for comment in data['comments']],
            'users': data['users'],
            'user_interactions': data['user_interactions'],
            'subreddits': data['subreddits'],
            'collection_timestamp': data['collection_timestamp'].isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, default=str)
        
        logger.info(f"Data saved to {filepath}")
    
    def _post_to_dict(self, post: RedditPost) -> Dict[str, Any]:
        """Convert RedditPost to dictionary"""
        return {
            'id': post.id,
            'title': post.title,
            'author': post.author,
            'subreddit': post.subreddit,
            'score': post.score,
            'num_comments': post.num_comments,
            'created_utc': post.created_utc.isoformat(),
            'selftext': post.selftext,
            'url': post.url,
            'is_self': post.is_self,
            'upvote_ratio': post.upvote_ratio,
            'flair': post.flair,
            'gilded': post.gilded
        }
    
    def _comment_to_dict(self, comment: RedditComment) -> Dict[str, Any]:
        """Convert RedditComment to dictionary"""
        return {
            'id': comment.id,
            'body': comment.body,
            'author': comment.author,
            'post_id': comment.post_id,
            'parent_id': comment.parent_id,
            'score': comment.score,
            'created_utc': comment.created_utc.isoformat(),
            'is_submitter': comment.is_submitter,
            'gilded': comment.gilded,
            'depth': comment.depth
        }


if __name__ == "__main__":
    # Example usage
    scraper = RedditScraper()
    
    # Collect data from political subreddits
    subreddits = ['politics', 'conspiracy', 'moderatepolitics', 'news']
    data = scraper.collect_subreddit_data(subreddits, limit=50)
    
    print(f"Collected {len(data['posts'])} posts and {len(data['comments'])} comments")
    
    # Save data
    scraper.save_data(data, 'reddit_data.json')
