"""
Visualization helpers for Streamlit web app.
Contains Plotly-based visualizations for explainability and metrics.
"""
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import Dict, List


def create_attention_heatmap(heatmap_data: Dict) -> go.Figure:
    """Create interactive heatmap of attention weights."""
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data['z'],
        x=heatmap_data['x'],
        y=heatmap_data['y'],
        colorscale='Viridis',
        hoverongaps=False,
        text=np.round(heatmap_data['z'], 2),
        texttemplate="%{text:.2f}",
        textfont={"size": 8},
        showscale=True
    ))
    
    fig.update_layout(
        title=heatmap_data.get('title', 'Attention Heatmap'),
        xaxis_title="Context Tokens",
        yaxis_title="Question Tokens",
        height=600,
        width=900,
        xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=8))
    )
    
    return fig


def create_token_importance_bar(importance_data: Dict) -> go.Figure:
    """Create bar chart of token importance scores."""
    tokens = importance_data['tokens'][:30]  # Show top 30 tokens
    scores = importance_data['scores'][:30]
    
    # Clean tokens for display
    clean_tokens = [t.replace('Ġ', ' ').replace('▁', ' ') for t in tokens]
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(range(len(clean_tokens))),
            y=scores,
            marker_color=scores,
            marker_colorscale='Bluered',
            hovertemplate='<b>Token:</b> %{customdata}<br><b>Importance:</b> %{y:.4f}<extra></extra>',
            customdata=clean_tokens
        )
    ])
    
    fig.update_layout(
        title='Token Importance Scores (Top 30)',
        xaxis_title='Token Position',
        yaxis_title='Importance Score',
        height=400,
        xaxis=dict(showticklabels=False),
        showlegend=False
    )
    
    return fig


def create_confusion_matrix(cm_data: List[List], labels: List[str]) -> go.Figure:
    """Create confusion matrix visualization."""
    fig = go.Figure(data=go.Heatmap(
        z=cm_data,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=cm_data,
        texttemplate="%{text}",
        textfont={"size": 14},
        showscale=True
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=400,
        width=500
    )
    
    return fig


def create_reliability_diagram(confidences: List[float], 
                               accuracies: List[float]) -> go.Figure:
    """Create reliability diagram for confidence calibration."""
    fig = go.Figure()
    
    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(color='red', dash='dash')
    ))
    
    # Actual calibration
    fig.add_trace(go.Scatter(
        x=confidences,
        y=accuracies,
        mode='lines+markers',
        name='Model Calibration',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title='Reliability Diagram: Confidence vs Accuracy',
        xaxis_title='Predicted Confidence',
        yaxis_title='Actual Accuracy',
        height=400,
        width=600,
        legend=dict(x=0.02, y=0.98)
    )
    
    return fig


def create_metrics_gauge(value: float, title: str, 
                         min_val: float = 0, max_val: float = 100) -> go.Figure:
    """Create gauge chart for single metric."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "gray"},
                {'range': [75, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=250, width=300)
    return fig


def create_error_distribution(error_types: Dict[str, int]) -> go.Figure:
    """Create pie chart of error type distribution."""
    labels = list(error_types.keys())
    values = list(error_types.values())
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.3,
        hovertemplate='<b>%{label}</b><br>Count: %{value}<extra></extra>'
    )])
    
    fig.update_layout(
        title='Error Type Distribution',
        height=400,
        width=500
    )
    
    return fig
