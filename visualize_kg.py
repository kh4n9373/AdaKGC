import networkx as nx
import matplotlib.pyplot as plt
import re
import os
import argparse
from collections import defaultdict

def parse_triplets(file_path):
    """Parse the knowledge graph triplets from the file."""
    triplets = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Extract all triplet lists
    pattern = r'\[\[(.*?)\]\]'
    triplet_groups = re.findall(pattern, content, re.DOTALL)
    
    for group in triplet_groups:
        # Split the group into individual triplets
        group_triplets = re.findall(r'\[\'(.*?)\', \'(.*?)\', \'(.*?)\'\]', group)
        triplets.extend(group_triplets)
        
    return triplets

def create_graph(triplets):
    """Create a networkx graph from the triplets."""
    G = nx.DiGraph()
    
    # Add nodes and edges
    for subject, predicate, obj in triplets:
        # Clean up entity names for better readability
        subject_clean = subject.replace('_', ' ')
        obj_clean = obj.replace('_', ' ')
        
        # Add nodes with attributes
        G.add_node(subject_clean, label=subject_clean)
        G.add_node(obj_clean, label=obj_clean)
        
        # Add edge with predicate as label
        G.add_edge(subject_clean, obj_clean, label=predicate)
    
    return G

def visualize_graph(G, output_path=None, show=True):
    """Visualize the graph using matplotlib."""
    plt.figure(figsize=(20, 16))
    
    # Use a layout that spreads nodes
    pos = nx.spring_layout(G, k=0.8, iterations=100)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color="lightblue", alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7, arrowsize=20)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    
    # Draw edge labels (predicates)
    edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.axis("off")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, format="png", dpi=300, bbox_inches="tight")
        print(f"Graph saved to {output_path}")
    
    if show:
        plt.show()

def create_subgraphs(G, max_nodes=20, output_dir=None):
    """Break down large graphs into smaller subgraphs for better visualization."""
    if len(G.nodes) <= max_nodes:
        return [G]
    
    # Group by first-level connections
    subgraphs = []
    remaining_nodes = set(G.nodes())
    
    while remaining_nodes:
        # Take a seed node
        seed = list(remaining_nodes)[0]
        
        # Get connected nodes (1-hop neighborhood)
        neighbors = set([seed]) | set(G.successors(seed)) | set(G.predecessors(seed))
        
        # Limit size of subgraph
        if len(neighbors) > max_nodes:
            neighbors = list(neighbors)[:max_nodes]
        
        # Create subgraph
        subgraph = G.subgraph(neighbors)
        subgraphs.append(subgraph)
        
        # Remove processed nodes
        remaining_nodes -= set(neighbors)
        
        # If there are remaining nodes but not enough to process
        if 0 < len(remaining_nodes) < 3:
            last_subgraph = G.subgraph(remaining_nodes)
            subgraphs.append(last_subgraph)
            break
    
    # Visualize each subgraph if output_dir provided
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if output_dir:
        for i, sg in enumerate(subgraphs):
            output_path = os.path.join(output_dir, f"subgraph_{i+1}.png")
            visualize_graph(sg, output_path=output_path, show=False)
    
    return subgraphs

def analyze_graph(G):
    """Analyze the graph and return some statistics."""
    stats = {
        "num_nodes": len(G.nodes),
        "num_edges": len(G.edges),
        "avg_degree": sum(dict(G.degree()).values()) / len(G.nodes),
        "predicates": defaultdict(int)
    }
    
    # Count predicate frequencies
    for _, _, data in G.edges(data=True):
        predicate = data.get("label", "")
        stats["predicates"][predicate] += 1
    
    # Sort predicates by frequency
    stats["top_predicates"] = sorted(
        stats["predicates"].items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:10]
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="Visualize knowledge graph from triplets")
    parser.add_argument("--input", type=str, default="edc/output/webnlg_target_alignment/iter0/canon_kg.txt",
                        help="Path to the triplets file")
    parser.add_argument("--output", type=str, default="edc/output/kg_visualization.png",
                        help="Path to save the visualization")
    parser.add_argument("--subgraphs", action="store_true", 
                        help="Create smaller subgraphs for better visualization")
    parser.add_argument("--max_nodes", type=int, default=20,
                        help="Maximum number of nodes per subgraph")
    parser.add_argument("--subgraph_dir", type=str, default="edc/output/subgraphs",
                        help="Directory to save subgraphs")
    
    args = parser.parse_args()
    
    # Parse triplets from file
    triplets = parse_triplets(args.input)
    print(f"Parsed {len(triplets)} triplets")
    
    # Create graph
    G = create_graph(triplets)
    print(f"Created graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
    
    # Analyze graph
    stats = analyze_graph(G)
    print("\nGraph Statistics:")
    print(f"Nodes: {stats['num_nodes']}")
    print(f"Edges: {stats['num_edges']}")
    print(f"Average degree: {stats['avg_degree']:.2f}")
    print("\nTop predicates:")
    for pred, count in stats["top_predicates"]:
        print(f"  {pred}: {count}")
    
    # Create subgraphs if specified
    if args.subgraphs:
        print(f"\nCreating subgraphs with max {args.max_nodes} nodes each...")
        subgraphs = create_subgraphs(G, max_nodes=args.max_nodes, output_dir=args.subgraph_dir)
        print(f"Created {len(subgraphs)} subgraphs")
    else:
        # Visualize the full graph
        visualize_graph(G, output_path=args.output)

if __name__ == "__main__":
    main() 