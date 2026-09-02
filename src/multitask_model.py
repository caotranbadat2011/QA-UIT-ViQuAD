"""
Multi-Task PhoBERT Model for Vietnamese Question Answering.

Implements a shared encoder with multiple task-specific heads:
1. Extractive QA (main task) - predicts answer span
2. Answerability Classification - determines if question is answerable
3. Answer Type Classification - categorizes answer type (PERSON, LOCATION, etc.)
4. Answer Length Prediction - estimates answer length in tokens
"""
import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Dict, Optional


class MultiTaskPhoBERT(nn.Module):
    """
    PhoBERT-based multi-task model for Question Answering.
    
    Architecture:
        Shared Encoder (PhoBERT base)
            ├── QA Head → start_logits, end_logits
            ├── Answerability Head → answerable/unanswerable
            ├── Answer Type Head → PERSON/LOCATION/DATE/NUMBER/OTHER
            └── Length Head → predicted answer length
    """
    
    def __init__(self, base_model_name: str = "vinai/phobert-base", 
                 dropout_rate: float = 0.1):
        super().__init__()
        
        # Shared encoder - PhoBERT base model
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.encoder.config.hidden_size  # 768 for base
        
        # Task 1: Extractive QA (Main Task)
        # Predicts start and end positions of answer span
        self.qa_dropout = nn.Dropout(dropout_rate)
        self.qa_head = nn.Linear(hidden_size, 2)  # start + end logits
        
        # Task 2: Answerability Classification (Auxiliary)
        # Binary classification: answerable vs unanswerable
        self.answerable_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 2)  # answerable / unanswerable
        )
        
        # Task 3: Answer Type Classification (Auxiliary)
        # 5 types: PERSON, LOCATION, DATE/TIME, NUMBER, OTHER
        self.type_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # 5 answer types
        )
        
        # Task 4: Answer Length Prediction (Auxiliary)
        # Regression: predict number of tokens in answer
        self.length_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # scalar length prediction
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize task-specific heads with proper initialization."""
        for module in [self.qa_head, self.answerable_head, self.type_head, self.length_head]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-task model.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            token_type_ids: Optional token type IDs
            position_ids: Optional position IDs
            
        Returns:
            Dictionary containing outputs from all heads:
            - start_logits: (batch_size, seq_len)
            - end_logits: (batch_size, seq_len)
            - answerable_logits: (batch_size, 2)
            - type_logits: (batch_size, 5)
            - length_pred: (batch_size, 1)
            - pooled_output: (batch_size, hidden_size)
            - last_hidden_state: (batch_size, seq_len, hidden_size)
        """
        # Encode with PhoBERT
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids
        )
        
        last_hidden_state = encoder_outputs.last_hidden_state  # (batch, seq_len, hidden)
        pooled_output = encoder_outputs.pooler_output  # (batch, hidden)
        
        # Apply dropout
        hidden_states = self.qa_dropout(last_hidden_state)
        
        # Task 1: QA Head
        qa_logits = self.qa_head(hidden_states)  # (batch, seq_len, 2)
        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # (batch, seq_len)
        end_logits = end_logits.squeeze(-1)  # (batch, seq_len)
        
        # Task 2: Answerability Head (use pooled output)
        answerable_logits = self.answerable_head(pooled_output)  # (batch, 2)
        
        # Task 3: Answer Type Head (use pooled output)
        type_logits = self.type_head(pooled_output)  # (batch, 5)
        
        # Task 4: Length Prediction Head (use pooled output)
        length_pred = self.length_head(pooled_output)  # (batch, 1)
        
        return {
            'start_logits': start_logits,
            'end_logits': end_logits,
            'answerable_logits': answerable_logits,
            'type_logits': type_logits,
            'length_pred': length_pred.squeeze(-1),
            'pooled_output': pooled_output,
            'last_hidden_state': last_hidden_state
        }
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Get parameter counts for each component."""
        total = sum(p.numel() for p in self.parameters())
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        head_params = total - encoder_params
        
        return {
            'total': total,
            'encoder': encoder_params,
            'heads': head_params,
            'trainable': total  # All params are trainable
        }


def compute_multitask_loss(outputs: Dict[str, torch.Tensor],
                           labels: Dict[str, torch.Tensor],
                           loss_weights: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
    """
    Compute weighted multi-task loss.
    
    Args:
        outputs: Model outputs from forward pass
        labels: Dictionary containing labels for each task:
            - start_positions: (batch_size,)
            - end_positions: (batch_size,)
            - answerable_labels: (batch_size,) - 0 or 1
            - type_labels: (batch_size,) - 0 to 4
            - length_labels: (batch_size,) - continuous values
        loss_weights: Weights for each task loss (default: balanced weights)
    
    Returns:
        Dictionary with individual losses and total loss
    """
    if loss_weights is None:
        loss_weights = {
            'qa': 0.6,           # Main task gets highest weight
            'answerable': 0.2,   # Auxiliary tasks
            'type': 0.1,
            'length': 0.1
        }
    
    # Task 1: QA Loss (CrossEntropy for start and end positions)
    ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
    
    start_logits = outputs['start_logits']
    end_logits = outputs['end_logits']
    start_positions = labels['start_positions']
    end_positions = labels['end_positions']
    
    start_loss = ce_loss(start_logits, start_positions)
    end_loss = ce_loss(end_logits, end_positions)
    qa_loss = (start_loss + end_loss) / 2.0
    
    # Task 2: Answerability Loss (CrossEntropy)
    if 'answerable_labels' in labels:
        answerable_loss = ce_loss(outputs['answerable_logits'], labels['answerable_labels'])
    else:
        answerable_loss = torch.tensor(0.0, device=start_logits.device)
    
    # Task 3: Answer Type Loss (CrossEntropy)
    if 'type_labels' in labels:
        type_loss = ce_loss(outputs['type_logits'], labels['type_labels'])
    else:
        type_loss = torch.tensor(0.0, device=start_logits.device)
    
    # Task 4: Length Prediction Loss (MSE for regression)
    if 'length_labels' in labels:
        length_mse = nn.MSELoss()
        length_loss = length_mse(outputs['length_pred'], labels['length_labels'].float())
    else:
        length_loss = torch.tensor(0.0, device=start_logits.device)
    
    # Weighted combination
    total_loss = (
        loss_weights['qa'] * qa_loss +
        loss_weights['answerable'] * answerable_loss +
        loss_weights['type'] * type_loss +
        loss_weights['length'] * length_loss
    )
    
    return {
        'total_loss': total_loss,
        'qa_loss': qa_loss,
        'start_loss': start_loss,
        'end_loss': end_loss,
        'answerable_loss': answerable_loss,
        'type_loss': type_loss,
        'length_loss': length_loss
    }


if __name__ == "__main__":
    # Test the model
    print("Testing MultiTaskPhoBERT...")
    
    model = MultiTaskPhoBERT("vinai/phobert-base")
    param_counts = model.get_num_parameters()
    
    print(f"\nModel Architecture:")
    print(f"  Total parameters: {param_counts['total']:,}")
    print(f"  Encoder parameters: {param_counts['encoder']:,}")
    print(f"  Head parameters: {param_counts['heads']:,}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    
    dummy_input = {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
        'attention_mask': torch.ones(batch_size, seq_len)
    }
    
    outputs = model(**dummy_input)
    
    print(f"\nForward pass outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # Test loss computation
    dummy_labels = {
        'start_positions': torch.tensor([3, 5]),
        'end_positions': torch.tensor([6, 8]),
        'answerable_labels': torch.tensor([1, 0]),
        'type_labels': torch.tensor([0, 2]),
        'length_labels': torch.tensor([4.0, 3.0])
    }
    
    losses = compute_multitask_loss(outputs, dummy_labels)
    
    print(f"\nLoss computation:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")
    
    print("\n✅ MultiTaskPhoBERT test passed!")
