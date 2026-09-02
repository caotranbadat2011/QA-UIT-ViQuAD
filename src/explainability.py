"""
Explainability module for Vietnamese Question Answering system.
Provides attention visualization and token importance analysis.
"""
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional


class AttentionVisualizer:
    """Extract and visualize attention weights from PhoBERT model."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def extract_attention_weights(self, question: str, context: str, 
                                  layer_idx: int = -1) -> Dict:
        """
        Extract attention weights for a specific layer.
        
        Args:
            question: Input question text
            context: Context paragraph
            layer_idx: Layer index to extract (-1 for last layer)
            
        Returns:
            Dictionary containing attention weights and token info
        """
        # Tokenize with attention output enabled
        inputs = self.tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=256,
            return_tensors="pt"
        )
        
        # Get sequence IDs to separate question from context
        sequence_ids = inputs.sequence_ids(0)
        
        # Forward pass with attention output
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        # Extract attention weights from specified layer
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        attentions = outputs.attentions[layer_idx][0].cpu().numpy()
        
        # Average across heads for simpler visualization
        avg_attention = attentions.mean(axis=0)
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Separate question and context tokens
        question_mask = [sid == 0 for sid in sequence_ids]
        context_mask = [sid == 1 for sid in sequence_ids]
        
        question_tokens = [t for t, m in zip(tokens, question_mask) if m and t not in ['[CLS]', '[SEP]', '<s>', '</s>']]
        context_tokens = [t for t, m in zip(tokens, context_mask) if m and t not in ['[CLS]', '[SEP]', '<s>', '</s>']]
        
        return {
            'attention_matrix': avg_attention,
            'tokens': tokens,
            'question_tokens': question_tokens,
            'context_tokens': context_tokens,
            'sequence_ids': sequence_ids.tolist(),
            'num_heads': attentions.shape[0],
            'layer_idx': layer_idx
        }
    
    def compute_question_context_attention(self, attention_data: Dict) -> np.ndarray:
        """
        Compute attention scores between question and context tokens.
        
        Returns:
            2D array of shape (len(question_tokens), len(context_tokens))
        """
        attn_matrix = attention_data['attention_matrix']
        seq_ids = attention_data['sequence_ids']
        
        # Find indices for question and context tokens
        q_indices = [i for i, sid in enumerate(seq_ids) if sid == 0]
        c_indices = [i for i, sid in enumerate(seq_ids) if sid == 1]
        
        if not q_indices or not c_indices:
            return np.zeros((1, 1))
        
        # Extract sub-matrix: question -> context attention
        q_c_attention = attn_matrix[np.ix_(q_indices, c_indices)]
        
        return q_c_attention
    
    def get_token_importance(self, question: str, context: str, 
                             answer: Optional[str] = None) -> Dict:
        """
        Compute importance score for each context token.
        
        Args:
            question: Input question
            context: Context text
            answer: Predicted answer (optional, for highlighting)
            
        Returns:
            Dictionary with token importance scores
        """
        # Extract multi-layer attention
        layers_to_use = [-1, -2, -3]  # Last 3 layers
        all_importances = []
        
        for layer_idx in layers_to_use:
            try:
                attn_data = self.extract_attention_weights(question, context, layer_idx)
                q_c_attn = self.compute_question_context_attention(attn_data)
                
                # Average attention received by each context token from question
                if q_c_attn.size > 0:
                    # Sum attention from all question tokens to each context token
                    context_importance = q_c_attn.mean(axis=0)
                    all_importances.append(context_importance)
            except Exception as e:
                continue
        
        if not all_importances:
            return {'tokens': [], 'scores': [], 'answer_positions': []}
        
        # Average across layers
        avg_importance = np.mean(all_importances, axis=0)
        
        # Normalize to [0, 1]
        if avg_importance.max() > avg_importance.min():
            normalized_scores = (avg_importance - avg_importance.min()) / (avg_importance.max() - avg_importance.min())
        else:
            normalized_scores = np.zeros_like(avg_importance)
        
        # Get context tokens
        inputs = self.tokenizer(question, context, truncation="only_second", max_length=256)
        sequence_ids = inputs.sequence_ids(0)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        context_tokens = [t for t, sid in zip(tokens, sequence_ids) if sid == 1 and t not in ['[CLS]', '[SEP]', '<s>', '</s>']]
        
        # Find answer positions
        answer_positions = []
        if answer and answer in context:
            answer_start = context.find(answer)
            answer_end = answer_start + len(answer)
            
            # Map character positions to token positions (approximate)
            char_pos = 0
            for idx, token in enumerate(context_tokens):
                clean_token = token.replace('Ġ', '').replace('▁', '')
                token_start = char_pos
                token_end = char_pos + len(clean_token)
                
                if token_start < answer_end and token_end > answer_start:
                    answer_positions.append(idx)
                
                char_pos += len(clean_token)
        
        return {
            'tokens': context_tokens[:len(normalized_scores)],
            'scores': normalized_scores.tolist(),
            'answer_positions': answer_positions,
            'question': question,
            'context': context
        }
    
    def generate_heatmap_data(self, question: str, context: str) -> Dict:
        """
        Generate data for heatmap visualization.
        
        Returns:
            Dictionary ready for Plotly heatmap
        """
        attn_data = self.extract_attention_weights(question, context)
        q_c_attention = self.compute_question_context_attention(attn_data)
        
        q_tokens = attn_data['question_tokens'][:q_c_attention.shape[0]]
        c_tokens = attn_data['context_tokens'][:q_c_attention.shape[1]]
        
        return {
            'z': q_c_attention.tolist(),
            'x': c_tokens,
            'y': q_tokens,
            'title': 'Attention Map: Question → Context'
        }


class TokenImportanceExplainer:
    """Explain predictions using token-level importance."""
    
    def __init__(self, visualizer: AttentionVisualizer):
        self.visualizer = visualizer
    
    def explain_prediction(self, question: str, context: str, 
                          answer: str, top_k: int = 10) -> Dict:
        """
        Explain why the model predicted this answer.
        
        Args:
            question: Input question
            context: Context text
            answer: Predicted answer
            top_k: Number of most important tokens to show
            
        Returns:
            Explanation dictionary
        """
        importance_data = self.visualizer.get_token_importance(question, context, answer)
        
        if not importance_data['tokens']:
            return {
                'explanation': 'Could not compute token importance.',
                'top_tokens': [],
                'answer_highlighted': answer
            }
        
        # Get top-k important tokens
        tokens = importance_data['tokens']
        scores = importance_data['scores']
        
        # Sort by importance
        indexed_scores = [(idx, score) for idx, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        top_tokens = []
        for idx, score in indexed_scores[:top_k]:
            top_tokens.append({
                'token': tokens[idx],
                'importance': round(score, 4),
                'position': idx
            })
        
        # Create highlighted context
        highlighted_context = context
        for token_info in top_tokens[:5]:  # Highlight top 5
            token = token_info['token'].replace('Ġ', ' ').replace('▁', ' ')
            if token.strip() and token.strip() in highlighted_context:
                highlighted_context = highlighted_context.replace(
                    token.strip(), 
                    f"**{token.strip()}**",
                    1
                )
        
        return {
            'explanation': f'The model focused on these tokens when answering "{question}":',
            'top_tokens': top_tokens,
            'answer_highlighted': answer,
            'highlighted_context': highlighted_context,
            'all_importances': importance_data
        }
