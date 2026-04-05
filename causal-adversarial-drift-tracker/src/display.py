import plotly.graph_objects as go
import networkx as nx
import os
from typing import Dict, Any, List, Optional
from .tracker import LiveDriftTracker
from .regulating import TruthRegulator

class DriftPlotter:
    """
    Plotly Drift Visualizer for CAD-TRACE.
    Creates an interactive HTML dashboard to visualize the Dynamic Interaction Graph (DIG).
    Color-codes nodes based on their Drift Level or Truth-Resilience score.
    """

    def __init__(self, tracker: LiveDriftTracker, regulator: TruthRegulator):
        """
        Initialize the DriftPlotter.

        Args:
            tracker: The LiveDriftTracker containing the DIG.
            regulator: The TruthRegulator for metrics.
        """
        self.tracker = tracker
        self.regulator = regulator

    def create_dashboard(self) -> go.Figure:
        """
        Generates an interactive Plotly figure of the DIG.
        Nodes are positioned using a hierarchical/tree layout or spring layout.
        Color indicates Truth-Resilience (Green = High, Red = Low).
        """
        G = self.tracker.dig
        
        # Use networkx layout
        # Graphviz 'dot' layout is better for DAGs/hierarchies but requires external libs.
        # We'll use spring_layout as a fallback.
        try:
            # Try to arrange by generation if possible
            pos = nx.spring_layout(G, k=0.5, iterations=50)
        except Exception:
            pos = {node: (0, 0) for node in G.nodes()}

        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            payload = self.tracker.get_payload(node)
            resilience = self.regulator.calculate_resilience_score(node)
            
            # Text for hover
            content_snippet = payload.content[:100] + "..." if payload else "N/A"
            drift_val = payload.drift_score if payload else 0.0
            
            hover_text = (
                f"Node ID: {node}<br>"
                f"Resilience: {resilience:.4f}<br>"
                f"Drift: {drift_val:.4f}<br>"
                f"Content: {content_snippet}"
            )
            node_text.append(hover_text)
            
            # Color mapping: 0.0 (red) to 1.0 (green)
            # Using Plotly's RdYlGn colorscale
            node_color.append(resilience)
            
            # Size nodes slightly by drift
            node_size.append(20)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='RdYlGn',
                reversescale=False,
                color=node_color,
                size=node_size,
                colorbar=dict(
                    thickness=15,
                    title='Truth-Resilience',
                    xanchor='left'
                ),
                line_width=2)
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        title=dict(text='CAD-TRACE: Dynamic Interaction Graph (DIG) - Drift Analysis', font=dict(size=16)),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="Green = High Resilience | Red = High Drift",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002 )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
        
        return fig

    def save_dashboard(self, filename: str = "visuals/drift_dashboard.html"):
        """
        Generates and saves the dashboard to an HTML file.
        """
        fig = self.create_dashboard()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        fig.write_html(filename)
        print(f"Dashboard saved to {filename}")

    def generate_report_summary(self) -> str:
        """
        Returns a string summary of the current drift state.
        """
        report = self.regulator.get_truth_report()
        summary = [
            "# CAD-TRACE Drift Summary",
            f"- **Total Nodes:** {report['total_nodes']}",
            f"- **Status:** {report['status'].upper()}",
            f"- **Drift Origin:** {report['drift_origin'] or 'None'}"
        ]
        
        if report['drifting_nodes']:
            summary.append("\n## Drifting Nodes Details:")
            for node in report['drifting_nodes']:
                summary.append(f"- `[{node['node_id']}]` Resilience: {node['score']:.4f}")
                
        return "\n".join(summary)
