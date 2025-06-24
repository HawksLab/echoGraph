"""
Web Interface for EchoGraph using Flask

This module provides a web-based interface for interacting with the
EchoGraph social echo chamber analyzer.
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import json
import tempfile
from datetime import datetime
from typing import Dict, Any
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_graph.graph_builder import GraphBuilder
from data_collection.reddit_scraper import RedditScraper
from analysis.echo_chamber_detector import EchoChamberDetector
from visualization.simple_visualizer import SimpleGraphVisualizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'echograph_secret_key_2025'

# Global variables for data persistence
current_graph = None
current_detector = None
current_visualizer = None
analysis_results = {}


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')


@app.route('/api/collect_data', methods=['POST'])
def collect_data():
    """API endpoint to collect Reddit data"""
    try:
        data = request.get_json()
        subreddits = data.get('subreddits', ['politics', 'conspiracy'])
        num_posts = data.get('num_posts', 100)
        
        logger.info(f"Collecting data from subreddits: {subreddits}")
        
        # Initialize scraper and collect data
        scraper = RedditScraper()
        reddit_data = scraper.collect_subreddit_data(subreddits, limit=num_posts)
        
        # Build knowledge graph
        global current_graph, current_detector, current_visualizer
        builder = GraphBuilder()
        current_graph = builder.build_from_reddit_data(reddit_data)
        
        # Initialize detector and visualizer
        current_detector = EchoChamberDetector(current_graph)
        current_visualizer = SimpleGraphVisualizer(current_graph)
        
        # Get basic statistics
        stats = current_graph.get_graph_statistics()
        
        return jsonify({
            'success': True,
            'message': 'Data collected successfully',
            'stats': stats,
            'collection_time': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error collecting data: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/analyze_echo_chambers', methods=['POST'])
def analyze_echo_chambers():
    """API endpoint to analyze echo chambers"""
    try:
        if not current_detector:
            return jsonify({
                'success': False,
                'error': 'No data loaded. Please collect data first.'
            }), 400
        
        data = request.get_json()
        min_size = data.get('min_size', 5)
        echo_threshold = data.get('echo_threshold', 0.7)
        
        logger.info("Analyzing echo chambers...")
        
        # Detect echo chambers
        echo_chambers = current_detector.detect_echo_chambers(min_size, echo_threshold)
        
        # Generate report
        report = current_detector.generate_echo_chamber_report()
        
        # Store results globally
        global analysis_results
        analysis_results = {
            'echo_chambers': echo_chambers,
            'report': report,
            'analysis_time': datetime.now().isoformat()
        }
        
        # Prepare response data
        chambers_data = []
        for chamber in echo_chambers:
            chambers_data.append({
                'id': chamber.community_id,
                'size': chamber.size,
                'echo_score': round(chamber.echo_score, 3),
                'density': round(chamber.density, 3),
                'homophily': round(chamber.homophily_score, 3),
                'content_diversity': round(chamber.content_diversity, 3),
                'polarization': round(chamber.polarization_index, 3),
                'subreddits': chamber.subreddits[:5],  # Top 5
                'topics': chamber.dominant_topics
            })
        
        return jsonify({
            'success': True,
            'echo_chambers': chambers_data,
            'summary': report['summary'],
            'network_metrics': report['network_metrics'],
            'recommendations': report['recommendations']
        })
        
    except Exception as e:
        logger.error(f"Error analyzing echo chambers: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/generate_visualization', methods=['POST'])
def generate_visualization():
    """API endpoint to generate visualizations"""
    try:
        if not current_visualizer:
            return jsonify({
                'success': False,
                'error': 'No data loaded. Please collect data first.'
            }), 400
        
        data = request.get_json()
        viz_type = data.get('type', 'network')
        
        logger.info(f"Generating {viz_type} visualization...")
        
        if viz_type == 'network':
            fig = current_visualizer.create_network_plot()
            html_content = current_visualizer.get_visualization_html(fig)
        elif viz_type == 'echo_chambers':
            if not analysis_results.get('echo_chambers'):
                return jsonify({
                    'success': False,
                    'error': 'Echo chamber analysis not performed. Please analyze first.'
                }), 400
            fig = current_visualizer.create_echo_chamber_analysis(analysis_results['echo_chambers'])
            html_content = current_visualizer.get_visualization_html(fig)
        elif viz_type == 'dashboard' or viz_type == 'statistics':
            fig = current_visualizer.create_network_statistics_plot()
            html_content = current_visualizer.get_visualization_html(fig)
        elif viz_type == 'all':
            # Return all visualizations
            visualizations = current_visualizer.create_all_visualizations()
            html_content = f"""
            <div class="visualization-grid">
                <div class="viz-section">
                    <h3>Network Graph</h3>
                    {visualizations.get('network', 'Error creating network visualization')}
                </div>
                <div class="viz-section">
                    <h3>Network Statistics</h3>
                    {visualizations.get('statistics', 'Error creating statistics visualization')}
                </div>
            </div>
            """
        else:
            return jsonify({
                'success': False,
                'error': f'Unknown visualization type: {viz_type}. Available: network, echo_chambers, statistics, all'
            }), 400
        
        return jsonify({
            'success': True,
            'html_content': html_content,
            'viz_type': viz_type
        })
        
    except Exception as e:
        logger.error(f"Error generating visualization: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/export_data', methods=['POST'])
def export_data():
    """API endpoint to export analysis results"""
    try:
        data = request.get_json()
        export_format = data.get('format', 'json')
        
        if not analysis_results:
            return jsonify({
                'success': False,
                'error': 'No analysis results to export.'
            }), 400
        
        # Create temporary file
        if export_format == 'json':
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                # Convert dataclasses to dictionaries for JSON serialization
                export_data = {
                    'summary': analysis_results['report']['summary'],
                    'network_metrics': analysis_results['report']['network_metrics'],
                    'echo_chambers': [],
                    'recommendations': analysis_results['report']['recommendations'],
                    'export_time': datetime.now().isoformat()
                }
                
                for chamber in analysis_results['echo_chambers']:
                    export_data['echo_chambers'].append({
                        'community_id': chamber.community_id,
                        'size': chamber.size,
                        'echo_score': chamber.echo_score,
                        'density': chamber.density,
                        'homophily_score': chamber.homophily_score,
                        'content_diversity': chamber.content_diversity,
                        'polarization_index': chamber.polarization_index,
                        'subreddits': chamber.subreddits,
                        'dominant_topics': chamber.dominant_topics,
                        'users': chamber.users
                    })
                
                json.dump(export_data, f, indent=2)
                temp_filename = f.name
        
        return send_file(
            temp_filename,
            as_attachment=True,
            download_name=f'echograph_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.{export_format}'
        )
        
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/get_graph_stats')
def get_graph_stats():
    """Get current graph statistics"""
    try:
        if not current_graph:
            return jsonify({
                'success': False,
                'error': 'No graph loaded.'
            }), 400
        
        stats = current_graph.get_graph_statistics()
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Error getting graph stats: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/documentation')
def documentation():
    """API documentation page"""
    return render_template('documentation.html')


@app.route('/examples')
def examples():
    """Examples and tutorials page"""
    return render_template('examples.html')


if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5002)
