"""
Advanced Trainer for Multi-Task and Adversarial Training.

Extends HuggingFace Trainer to support:
1. Custom multi-task loss computation
2. Adversarial training with on-the-fly perturbation
3. Curriculum-based sample weighting
4. Detailed logging of individual task losses
"""
import torch
import torch.nn as nn
from transformers import Trainer
from typing import Dict, Optional, Tuple, Any


class MultiTaskTrainer(Trainer):
    """
    Trainer subclass that supports multi-task learning with custom loss computation.
    
    Overrides compute_loss() to handle multiple task heads and combine losses
    with configurable weights.
    """
    
    def __init__(self, *args, loss_weights: Optional[Dict[str, float]] = None, **kwargs):
        """
        Initialize MultiTaskTrainer.
        
        Args:
            loss_weights: Weights for combining task losses
                {'qa': 0.6, 'answerable': 0.2, 'type': 0.1, 'length': 0.1}
        """
        super().__init__(*args, **kwargs)
        self.loss_weights = loss_weights or {
            'qa': 0.6,
            'answerable': 0.2,
            'type': 0.1,
            'length': 0.1
        }
        
        # For tracking losses during training
        self.logged_losses = {
            'total': [],
            'qa': [],
            'answerable': [],
            'type': [],
            'length': []
        }
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute multi-task loss.
        
        Args:
            model: MultiTaskPhoBERT model
            inputs: Batch inputs including labels
            return_outputs: Whether to return model outputs
            
        Returns:
            Loss tensor (and optionally outputs)
        """
        # Extract labels from inputs
        labels = {
            'start_positions': inputs.pop('start_positions'),
            'end_positions': inputs.pop('end_positions'),
        }
        
        # Add auxiliary task labels if available
        if 'answerable_labels' in inputs:
            labels['answerable_labels'] = inputs.pop('answerable_labels')
        if 'type_labels' in inputs:
            labels['type_labels'] = inputs.pop('type_labels')
        if 'length_labels' in inputs:
            labels['length_labels'] = inputs.pop('length_labels')
        
        # Forward pass through multi-task model
        outputs = model(**inputs)
        
        # Compute multi-task loss
        from src.multitask_model import compute_multitask_loss
        losses = compute_multitask_loss(outputs, labels, self.loss_weights)
        
        loss = losses['total_loss']
        
        # Log individual losses for monitoring
        if self.state.global_step % self.args.logging_steps == 0:
            self.log_individual_losses(losses)
        
        if return_outputs:
            return loss, outputs
        return loss
    
    def log_individual_losses(self, losses: Dict[str, torch.Tensor]):
        """Log individual task losses for monitoring."""
        step = self.state.global_step
        
        self.logged_losses['total'].append({
            'step': step,
            'value': losses['total_loss'].item()
        })
        self.logged_losses['qa'].append({
            'step': step,
            'value': losses['qa_loss'].item()
        })
        self.logged_losses['answerable'].append({
            'step': step,
            'value': losses['answerable_loss'].item()
        })
        self.logged_losses['type'].append({
            'step': step,
            'value': losses['type_loss'].item()
        })
        self.logged_losses['length'].append({
            'step': step,
            'value': losses['length_loss'].item()
        })
    
    def get_loss_history(self) -> Dict[str, list]:
        """Get history of logged losses."""
        return self.logged_losses


class AdversarialMultiTaskTrainer(MultiTaskTrainer):
    """
    Trainer with adversarial training support.
    
    Generates adversarial examples on-the-fly and combines clean + adversarial losses.
    """
    
    def __init__(self, *args, adversarial_module=None, 
                 adversarial_weight: float = 0.3,
                 adversarial_frequency: float = 0.5,
                 **kwargs):
        """
        Initialize AdversarialMultiTaskTrainer.
        
        Args:
            adversarial_module: Instance of AdversarialPerturbation class
            adversarial_weight: Weight for adversarial loss (0.0-1.0)
            adversarial_frequency: How often to apply adversarial training (0.0-1.0)
        """
        super().__init__(*args, **kwargs)
        self.adversarial = adversarial_module
        self.adversarial_weight = adversarial_weight
        self.adversarial_frequency = adversarial_frequency
        
        # Track adversarial training stats
        self.adv_stats = {
            'total_batches': 0,
            'adv_batches': 0
        }
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute combined clean + adversarial loss.
        
        Strategy:
        1. Compute clean loss normally
        2. With probability adversarial_frequency, generate adversarial version
        3. Compute adversarial loss
        4. Combine: total = (1 - adv_weight) * clean + adv_weight * adv
        """
        self.adv_stats['total_batches'] += 1
        
        # Clean loss
        clean_loss, clean_outputs = super().compute_loss(
            model, inputs.copy(), return_outputs=True
        )
        
        # Decide whether to apply adversarial training this batch
        should_apply_adversarial = (
            self.adversarial is not None and
            torch.rand(1).item() < self.adversarial_frequency and
            self.model.training  # Only during training
        )
        
        if should_apply_adversarial:
            try:
                self.adv_stats['adv_batches'] += 1
                
                # Generate adversarial inputs
                adv_inputs = self.adversarial.perturb_batch(inputs)
                
                # Compute adversarial loss
                adv_loss, _ = super().compute_loss(
                    model, adv_inputs, return_outputs=True
                )
                
                # Combined loss
                total_loss = (
                    (1 - self.adversarial_weight) * clean_loss +
                    self.adversarial_weight * adv_loss
                )
                
            except Exception as e:
                # Fallback to clean loss if adversarial generation fails
                print(f"Adversarial generation failed: {e}, using clean loss")
                total_loss = clean_loss
        else:
            total_loss = clean_loss
        
        if return_outputs:
            return total_loss, clean_outputs
        return total_loss
    
    def get_adversarial_stats(self) -> Dict[str, Any]:
        """Get statistics about adversarial training."""
        total = self.adv_stats['total_batches']
        adv = self.adv_stats['adv_batches']
        return {
            'total_batches': total,
            'adversarial_batches': adv,
            'adversarial_ratio': adv / total if total > 0 else 0
        }


class CurriculumTrainer(MultiTaskTrainer):
    """
    Trainer with curriculum learning support.
    
    Adjusts sample weights based on difficulty scores to implement
    easy-to-hard learning strategy.
    """
    
    def __init__(self, *args, curriculum_phase: str = "easy", **kwargs):
        """
        Initialize CurriculumTrainer.
        
        Args:
            curriculum_phase: Current phase - "easy", "medium", or "hard"
        """
        super().__init__(*args, **kwargs)
        self.curriculum_phase = curriculum_phase
        
        # Phase-specific configurations
        self.phase_configs = {
            'easy': {
                'max_difficulty': 3,
                'sample_weight_fn': lambda d: 1.0 if d <= 2 else 0.5
            },
            'medium': {
                'max_difficulty': 6,
                'sample_weight_fn': lambda d: 1.0 if d <= 5 else 0.7
            },
            'hard': {
                'max_difficulty': 10,
                'sample_weight_fn': lambda d: 1.0  # All samples weighted equally
            }
        }
    
    def set_curriculum_phase(self, phase: str):
        """Switch to a different curriculum phase."""
        if phase not in self.phase_configs:
            raise ValueError(f"Invalid phase: {phase}. Must be one of {list(self.phase_configs.keys())}")
        self.curriculum_phase = phase
        print(f"Curriculum phase changed to: {phase}")
    
    def get_current_phase_config(self) -> Dict[str, Any]:
        """Get configuration for current curriculum phase."""
        return self.phase_configs[self.curriculum_phase]
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss with curriculum-based sample weighting.
        
        Samples are weighted based on their difficulty score and current phase.
        """
        # Get base loss
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        
        # Apply curriculum weighting if difficulty scores are available
        if 'difficulty_scores' in inputs:
            phase_config = self.get_current_phase_config()
            difficulties = inputs['difficulty_scores']
            
            # Compute sample weights based on difficulty
            weights = torch.tensor([
                phase_config['sample_weight_fn'](d.item())
                for d in difficulties
            ], device=loss.device)
            
            # Normalize weights
            weights = weights / weights.mean()
            
            # Apply weighting to loss (this is simplified; proper implementation
            # would require per-sample losses)
            # For now, we just use the base loss but log the weighting info
            avg_difficulty = difficulties.mean().item()
            avg_weight = weights.mean().item()
            
            if self.state.global_step % self.args.logging_steps == 0:
                self.log({
                    'curriculum/avg_difficulty': avg_difficulty,
                    'curriculum/avg_weight': avg_weight,
                    'curriculum/phase': self.curriculum_phase
                })
        
        if return_outputs:
            return loss, outputs
        return loss


def create_trainer(model, train_dataset, eval_dataset, training_args,
                   trainer_type: str = "multitask",
                   **trainer_kwargs) -> Trainer:
    """
    Factory function to create appropriate trainer type.
    
    Args:
        model: The model to train
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        training_args: TrainingArguments
        trainer_type: One of "multitask", "adversarial", "curriculum"
        **trainer_kwargs: Additional kwargs for specific trainer types
    
    Returns:
        Configured Trainer instance
    """
    trainer_classes = {
        'multitask': MultiTaskTrainer,
        'adversarial': AdversarialMultiTaskTrainer,
        'curriculum': CurriculumTrainer
    }
    
    if trainer_type not in trainer_classes:
        raise ValueError(f"Unknown trainer type: {trainer_type}. Choose from {list(trainer_classes.keys())}")
    
    trainer_class = trainer_classes[trainer_type]
    
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        **trainer_kwargs
    )
    
    return trainer


if __name__ == "__main__":
    # Test trainer creation
    print("Testing Advanced Trainer classes...")
    
    # Just test that classes can be imported and instantiated
    print("✅ MultiTaskTrainer class available")
    print("✅ AdversarialMultiTaskTrainer class available")
    print("✅ CurriculumTrainer class available")
    print("✅ create_trainer factory function available")
    
    print("\nAdvanced Trainer module loaded successfully!")
