# EchoGraph - Social Echo Chamber Analyzer

A comprehensive tool for analyzing social media echo chambers and misinformation pathways using knowledge graphs.

## 🎯 Project Overview

EchoGraph analyzes social media data (Reddit/Twitter) to:
- Build knowledge graphs from user interactions
- Detect echo chambers and filter bubbles
- Identify misinformation pathways
- Visualize community structures
- Provide early warning for coordinated misinformation campaigns

## Demo:
https://drive.google.com/file/d/12CCHeLH6UgBxAytda2WyfnTBSl2OUAdK/view?usp=sharing
## 🏗️ Architecture

```
echoGraph/
├── src/
│   ├── knowledge_graph/     # Custom KG implementation
│   ├── data_collection/     # Reddit/Twitter scrapers
│   ├── analysis/           # Echo chamber & misinformation detection
│   ├── visualization/      # Graph visualization
│   └── web_interface/      # Flask web app
├── data/                   # Sample datasets
├── models/                 # Trained models
├── tests/                  # Unit tests
└── notebooks/              # Jupyter analysis notebooks
```

## 🚀 Features

### Core Knowledge Graph
- Custom graph implementation with nodes and edges
- Support for user, post, and content entities
- Relationship modeling (follows, shares, comments)
- Graph traversal and analysis algorithms

### Echo Chamber Detection
- Community detection using custom algorithms
- Echo chamber scoring based on content diversity
- Polarization measurement
- Filter bubble identification

### Misinformation Analysis
- Content similarity analysis
- Coordinated behavior detection
- Information pathway tracing
- Early warning system

### Visualization
- Interactive network graphs
- Community highlighting
- Temporal analysis views
- Dashboard with metrics

## 📊 Datasets

The project works with:
- Reddit post and comment data
- User interaction networks
- News article URLs and metadata
- Optional fact-checking labels

## 🛠️ Installation

```bash
# Clone the repository
git clone <https://github.com/HawksLab/echoGraph>
cd echoGraph

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python src/main.py
```

## 🎮 Usage

### 1. Data Collection
```python
from src.data_collection.reddit_scraper import RedditScraper

scraper = RedditScraper()
data = scraper.collect_subreddit_data(['politics', 'conspiracy'], days=30)
```

### 2. Build Knowledge Graph
```python
from src.knowledge_graph.graph import KnowledgeGraph

kg = KnowledgeGraph()
kg.build_from_reddit_data(data)
```

### 3. Analyze Echo Chambers
```python
from src.analysis.echo_chamber_detector import EchoChamberDetector

detector = EchoChamberDetector(kg)
chambers = detector.detect_echo_chambers()
scores = detector.calculate_echo_scores()
```

### 4. Visualize Results
```python
from src.visualization.graph_visualizer import GraphVisualizer

viz = GraphVisualizer(kg)
viz.create_interactive_plot(chambers)
viz.save_html("echo_chambers.html")
```

## 📈 Analysis Methods

### Echo Chamber Detection
- **Modularity Analysis**: Identifies tightly connected communities
- **Content Diversity Score**: Measures information variety within groups
- **Cross-Group Interaction**: Analyzes inter-community connections
- **Temporal Dynamics**: Tracks chamber formation over time

### Misinformation Detection
- **Content Similarity**: Identifies repeated/coordinated content
- **Propagation Patterns**: Traces information spread pathways
- **User Behavior Analysis**: Detects suspicious coordination
- **Network Anomalies**: Identifies unusual connection patterns

## 🧪 Algorithms Implemented

1. **Custom Graph Algorithms**
   - Breadth-First Search (BFS)
   - Depth-First Search (DFS)
   - Shortest Path (Dijkstra's)
   - Community Detection (Louvain-inspired)

2. **Echo Chamber Metrics**
   - Polarization Index
   - Diversity Score
   - Isolation Coefficient
   - Homophily Measure

3. **Misinformation Detection**
   - Content Similarity (TF-IDF + Cosine)
   - Behavioral Anomaly Detection
   - Cascade Analysis
   - Coordination Scoring

## 📊 Sample Results

The system provides:
- Interactive network visualizations
- Echo chamber reports with scores
- Misinformation pathway maps
- Temporal analysis charts
- User and content analytics

## 🔬 Research Applications

This project can be used for:
- Academic research on social media dynamics
- Digital humanities studies
- Computational social science
- Media literacy education
- Platform policy research

