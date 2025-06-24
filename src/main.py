"""
Main Entry Point for EchoGraph Application

This module provides the main interface for running EchoGraph analysis
either as a command-line tool or web application.
"""

import argparse
import logging
import sys
import os
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from knowledge_graph.graph_builder import GraphBuilder
from data_collection.reddit_scraper import RedditScraper
from analysis.echo_chamber_detector import EchoChamberDetector
from visualization.graph_visualizer import GraphVisualizer
from web_interface.app import app

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EchoGraphAnalyzer:
    """
    Main EchoGraph analyzer class that coordinates all components
    """
    
    def __init__(self):
        self.scraper = None
        self.graph_builder = None
        self.knowledge_graph = None
        self.detector = None
        self.visualizer = None
        self.results = {}
    
    def run_full_analysis(self, subreddits: list, num_posts: int = 100, 
                         output_dir: str = "output") -> dict:
        """
        Run complete echo chamber analysis pipeline
        
        Args:
            subreddits: List of subreddit names to analyze
            num_posts: Number of posts to collect per subreddit
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info("Starting full EchoGraph analysis...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Data Collection
        logger.info("Step 1: Collecting data...")
        self.scraper = RedditScraper()
        reddit_data = self.scraper.collect_subreddit_data(subreddits, limit=num_posts)
        
        # Save raw data
        data_file = os.path.join(output_dir, "reddit_data.json")
        self.scraper.save_data(reddit_data, data_file)
        logger.info(f"Raw data saved to {data_file}")
        
        # Step 2: Knowledge Graph Construction
        logger.info("Step 2: Building knowledge graph...")
        self.graph_builder = GraphBuilder()
        self.knowledge_graph = self.graph_builder.build_from_reddit_data(reddit_data)
        
        # Save graph
        graph_file = os.path.join(output_dir, "knowledge_graph.pkl")
        self.knowledge_graph.save_to_file(graph_file)
        logger.info(f"Knowledge graph saved to {graph_file}")
        
        # Step 3: Echo Chamber Detection
        logger.info("Step 3: Detecting echo chambers...")
        self.detector = EchoChamberDetector(self.knowledge_graph)
        echo_chambers = self.detector.detect_echo_chambers()
        
        # Generate comprehensive report
        report = self.detector.generate_echo_chamber_report()
        
        # Step 4: Visualizations
        logger.info("Step 4: Creating visualizations...")
        self.visualizer = GraphVisualizer(self.knowledge_graph)
        
        # Create different types of visualizations
        visualizations = {
            'network_overview': self.visualizer.create_network_visualization(),
            'echo_chambers': self.visualizer.create_echo_chamber_visualization(echo_chambers),
            'dashboard': self.visualizer.create_echo_chamber_dashboard(echo_chambers, self.detector),
            'temporal': self.visualizer.create_temporal_analysis()
        }
        
        # Save visualizations
        for viz_name, fig in visualizations.items():
            viz_file = os.path.join(output_dir, f"{viz_name}.html")
            self.visualizer.save_visualization(fig, viz_file)
            logger.info(f"Visualization saved to {viz_file}")
        
        # Step 5: Save Analysis Report
        report_file = os.path.join(output_dir, "echo_chamber_report.json")
        import json
        
        # Convert dataclasses to dictionaries for JSON serialization
        serializable_report = {
            'summary': report['summary'],
            'network_metrics': report['network_metrics'],
            'recommendations': report['recommendations'],
            'echo_chambers': [],
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        for chamber in echo_chambers:
            serializable_report['echo_chambers'].append({
                'community_id': chamber.community_id,
                'size': chamber.size,
                'echo_score': chamber.echo_score,
                'density': chamber.density,
                'homophily_score': chamber.homophily_score,
                'content_diversity': chamber.content_diversity,
                'polarization_index': chamber.polarization_index,
                'subreddits': chamber.subreddits,
                'dominant_topics': chamber.dominant_topics,
                'users': chamber.users[:10]  # Limit for privacy
            })
        
        with open(report_file, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        
        logger.info(f"Analysis report saved to {report_file}")
        
        # Compile results
        self.results = {
            'subreddits_analyzed': subreddits,
            'posts_collected': len(reddit_data['posts']),
            'graph_stats': self.knowledge_graph.get_graph_statistics(),
            'echo_chambers_detected': len(echo_chambers),
            'report': serializable_report,
            'output_directory': output_dir,
            'files_created': [
                data_file, graph_file, report_file,
                *[os.path.join(output_dir, f"{name}.html") for name in visualizations.keys()]
            ]
        }
        
        logger.info("EchoGraph analysis completed successfully!")
        return self.results
    
    def print_summary(self):
        """Print analysis summary to console"""
        if not self.results:
            print("No analysis results available. Run analysis first.")
            return
        
        print("\n" + "="*60)
        print("ECHOGRAPH ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Subreddits Analyzed: {', '.join(self.results['subreddits_analyzed'])}")
        print(f"Posts Collected: {self.results['posts_collected']}")
        print(f"Graph Nodes: {self.results['graph_stats']['num_nodes']}")
        print(f"Graph Edges: {self.results['graph_stats']['num_edges']}")
        print(f"Echo Chambers Detected: {self.results['echo_chambers_detected']}")
        
        print(f"\nOutput Directory: {self.results['output_directory']}")
        print("Files Created:")
        for file_path in self.results['files_created']:
            print(f"  - {file_path}")
        
        # Print echo chamber details
        if self.results['echo_chambers_detected'] > 0:
            print(f"\nEcho Chamber Details:")
            for chamber in self.results['report']['echo_chambers']:
                print(f"  Chamber {chamber['community_id']}:")
                print(f"    Size: {chamber['size']} users")
                print(f"    Echo Score: {chamber['echo_score']:.3f}")
                print(f"    Density: {chamber['density']:.3f}")
                print(f"    Content Diversity: {chamber['content_diversity']:.3f}")
                print(f"    Topics: {', '.join(chamber['dominant_topics']) if chamber['dominant_topics'] else 'None'}")
                print()
        
        # Print recommendations
        print("Recommendations:")
        for rec in self.results['report']['recommendations']:
            print(f"  • {rec}")
        
        print("\n" + "="*60)


def main():
    """Main command-line interface"""
    parser = argparse.ArgumentParser(description='EchoGraph - Social Echo Chamber Analyzer')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Run echo chamber analysis')
    analyze_parser.add_argument('--subreddits', nargs='+', 
                               default=['politics', 'conspiracy', 'moderatepolitics'],
                               help='Subreddits to analyze')
    analyze_parser.add_argument('--posts', type=int, default=100,
                               help='Number of posts to collect per subreddit')
    analyze_parser.add_argument('--output', default='output',
                               help='Output directory for results')
    
    # Web interface command
    web_parser = subparsers.add_parser('web', help='Start web interface')
    web_parser.add_argument('--host', default='0.0.0.0', help='Host address')
    web_parser.add_argument('--port', type=int, default=5000, help='Port number')
    web_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demo analysis with sample data')
    demo_parser.add_argument('--output', default='demo_output',
                            help='Output directory for demo results')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        # Run analysis
        analyzer = EchoGraphAnalyzer()
        results = analyzer.run_full_analysis(
            subreddits=args.subreddits,
            num_posts=args.posts,
            output_dir=args.output
        )
        analyzer.print_summary()
        
    elif args.command == 'web':
        # Start web interface
        print(f"Starting EchoGraph web interface at http://{args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=args.debug)
        
    elif args.command == 'demo':
        # Run demo analysis
        print("Running EchoGraph demo with sample data...")
        analyzer = EchoGraphAnalyzer()
        
        # Use demo data
        demo_subreddits = ['politics', 'conspiracy', 'moderatepolitics', 'news']
        results = analyzer.run_full_analysis(
            subreddits=demo_subreddits,
            num_posts=50,  # Smaller for demo
            output_dir=args.output
        )
        
        analyzer.print_summary()
        print(f"\nDemo completed! Check the '{args.output}' directory for results.")
        print("Open the HTML files in a web browser to see the visualizations.")
        
    else:
        # Show help if no command provided
        parser.print_help()


if __name__ == "__main__":
    main()
