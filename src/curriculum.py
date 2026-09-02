"""
Curriculum Learning for Vietnamese Question Answering.

Implements difficulty-based sample ordering to train from easy to hard,
improving convergence and final performance.
"""
import random
from typing import Dict, List, Tuple, Optional


class CurriculumScheduler:
    """
    Schedule training samples by difficulty level.
    
    Strategy:
    1. Compute difficulty score for each sample (0-10)
    2. Split into phases: Easy (0-3), Medium (4-6), Hard (7-10)
    3. Train sequentially through phases
    """
    
    def __init__(self):
        # Difficulty scoring weights
        self.weights = {
            'context_length': 3,      # Max contribution: 3 points
            'question_complexity': 2,  # Max: 2 points
            'answer_position': 2,      # Max: 2 points
            'negation': 2,             # Max: 2 points
            'impossible': 3            # Max: 3 points (cap at 10)
        }
    
    def compute_difficulty(self, sample: Dict) -> int:
        """
        Compute difficulty score for a single sample (0-10).
        
        Factors:
        1. Context length - longer contexts are harder
        2. Question complexity - longer/more complex questions
        3. Answer position - answers deep in text are harder to find
        4. Negation - questions with negation are trickier
        5. Impossible - unanswerable questions are hardest
        
        Args:
            sample: Dictionary with keys: question, context, answer_start, is_impossible
            
        Returns:
            Difficulty score from 0 (easiest) to 10 (hardest)
        """
        score = 0
        
        # Factor 1: Context length
        context_words = len(sample.get('context', '').split())
        if context_words > 200:
            score += 3
        elif context_words > 100:
            score += 2
        elif context_words > 50:
            score += 1
        
        # Factor 2: Question complexity
        question = sample.get('question', '')
        question_words = len(question.split())
        if question_words > 15:
            score += 2
        elif question_words > 10:
            score += 1
        
        # Check for implicit questions (no question mark)
        if '?' not in question and len(question) > 5:
            score += 1
        
        # Factor 3: Answer position (if available)
        answer_start = sample.get('answer_start', -1)
        if answer_start > 0:
            # Normalize by context length
            context_len = len(sample.get('context', ''))
            if context_len > 0:
                relative_pos = answer_start / context_len
                if relative_pos > 0.7:  # Answer is in last 30% of text
                    score += 2
                elif relative_pos > 0.4:  # Answer is in middle
                    score += 1
        
        # Factor 4: Negation in question
        negation_words = ['không', 'chưa', 'chẳng', 'đâu', 'không phải', 'chưa từng']
        question_lower = question.lower()
        if any(neg in question_lower for neg in negation_words):
            score += 2
        
        # Factor 5: Is impossible/unanswerable
        if sample.get('is_impossible', False):
            score += 3
        
        # Cap at 10
        return min(score, 10)
    
    def score_dataset(self, dataset) -> List[Tuple]:
        """
        Compute difficulty scores for entire dataset.
        
        Args:
            dataset: HuggingFace Dataset or list of dicts
            
        Returns:
            List of (sample, difficulty_score) tuples
        """
        scored_samples = []
        
        # Handle both Dataset and list formats
        if hasattr(dataset, '__len__') and hasattr(dataset, '__getitem__'):
            for i in range(len(dataset)):
                sample = dataset[i] if isinstance(dataset, list) else dict(dataset[i])
                difficulty = self.compute_difficulty(sample)
                scored_samples.append((sample, difficulty))
        else:
            # Assume it's already a list of dicts
            for sample in dataset:
                difficulty = self.compute_difficulty(sample)
                scored_samples.append((sample, difficulty))
        
        return scored_samples
    
    def split_by_difficulty(self, scored_samples: List[Tuple]) -> Dict[str, List]:
        """
        Split scored samples into curriculum phases.
        
        Args:
            scored_samples: List of (sample, difficulty) tuples
            
        Returns:
            Dictionary with keys: 'easy', 'medium', 'hard'
        """
        phases = {
            'easy': [],    # Difficulty 0-3
            'medium': [],  # Difficulty 4-6
            'hard': []     # Difficulty 7-10
        }
        
        for sample, difficulty in scored_samples:
            if difficulty <= 3:
                phases['easy'].append((sample, difficulty))
            elif difficulty <= 6:
                phases['medium'].append((sample, difficulty))
            else:
                phases['hard'].append((sample, difficulty))
        
        # Shuffle within each phase
        for phase in phases:
            random.shuffle(phases[phase])
        
        return phases
    
    def get_phase_samples(self, scored_samples: List[Tuple], 
                          phase: str) -> List:
        """
        Get samples for a specific curriculum phase.
        
        Args:
            scored_samples: List of (sample, difficulty) tuples
            phase: One of 'easy', 'medium', 'hard', 'all'
            
        Returns:
            List of samples (without difficulty scores)
        """
        if phase == 'all':
            return [s for s, _ in scored_samples]
        
        thresholds = {
            'easy': (0, 3),
            'medium': (4, 6),
            'hard': (7, 10)
        }
        
        if phase not in thresholds:
            raise ValueError(f"Invalid phase: {phase}. Choose from {list(thresholds.keys())}")
        
        min_d, max_d = thresholds[phase]
        samples = [s for s, d in scored_samples if min_d <= d <= max_d]
        random.shuffle(samples)
        
        return samples
    
    def get_curriculum_schedule(self, total_epochs: int = 6) -> List[Dict]:
        """
        Generate curriculum training schedule.
        
        Args:
            total_epochs: Total number of training epochs
            
        Returns:
            List of phase configurations for each epoch
        """
        schedule = []
        
        if total_epochs <= 2:
            # Too few epochs for full curriculum, just use all data
            schedule = [{'phase': 'all', 'epochs': total_epochs}]
        elif total_epochs <= 4:
            # Simplified curriculum: easy → all
            schedule = [
                {'phase': 'easy', 'epochs': 1},
                {'phase': 'all', 'epochs': total_epochs - 1}
            ]
        else:
            # Full curriculum: easy → medium → hard/all
            easy_epochs = max(1, total_epochs // 3)
            medium_epochs = max(1, total_epochs // 3)
            remaining = total_epochs - easy_epochs - medium_epochs
            
            schedule = [
                {'phase': 'easy', 'epochs': easy_epochs},
                {'phase': 'medium', 'epochs': medium_epochs},
                {'phase': 'all', 'epochs': remaining}
            ]
        
        return schedule
    
    def get_difficulty_distribution(self, scored_samples: List[Tuple]) -> Dict[int, int]:
        """
        Get distribution of difficulty scores.
        
        Returns:
            Dictionary mapping difficulty score to count
        """
        distribution = {}
        for _, difficulty in scored_samples:
            distribution[difficulty] = distribution.get(difficulty, 0) + 1
        
        return dict(sorted(distribution.items()))
    
    def print_summary(self, scored_samples: List[Tuple]):
        """Print summary of curriculum dataset."""
        print("\n" + "="*60)
        print("CURRICULUM LEARNING DATASET SUMMARY")
        print("="*60)
        
        total = len(scored_samples)
        difficulties = [d for _, d in scored_samples]
        
        print(f"\nTotal samples: {total}")
        print(f"Difficulty range: {min(difficulties)} - {max(difficulties)}")
        print(f"Mean difficulty: {sum(difficulties)/len(difficulties):.2f}")
        
        # Phase breakdown
        phases = self.split_by_difficulty(scored_samples)
        
        print(f"\nPhase Breakdown:")
        print(f"  Easy   (0-3):   {len(phases['easy']):4d} samples ({len(phases['easy'])/total*100:.1f}%)")
        print(f"  Medium (4-6):   {len(phases['medium']):4d} samples ({len(phases['medium'])/total*100:.1f}%)")
        print(f"  Hard   (7-10):  {len(phases['hard']):4d} samples ({len(phases['hard'])/total*100:.1f}%)")
        
        # Detailed distribution
        dist = self.get_difficulty_distribution(scored_samples)
        print(f"\nDetailed Distribution:")
        for score, count in sorted(dist.items()):
            bar = "█" * (count // 10) if count >= 10 else "▓"
            print(f"  Score {score:2d}: {count:4d} {bar}")
        
        # Training schedule
        schedule = self.get_curriculum_schedule()
        print(f"\nRecommended Training Schedule:")
        for i, phase_config in enumerate(schedule):
            print(f"  Phase {i+1}: {phase_config['phase']:8s} for {phase_config['epochs']} epoch(s)")
        
        print("="*60 + "\n")


def create_curriculum_datasets(dataset, scheduler: CurriculumScheduler = None):
    """
    Convenience function to create curriculum datasets from raw dataset.
    
    Args:
        dataset: Raw dataset (HuggingFace Dataset or list)
        scheduler: Optional CurriculumScheduler instance
        
    Returns:
        Tuple of (scored_samples, phases_dict, schedule)
    """
    if scheduler is None:
        scheduler = CurriculumScheduler()
    
    # Score all samples
    scored_samples = scheduler.score_dataset(dataset)
    
    # Split into phases
    phases = scheduler.split_by_difficulty(scored_samples)
    
    # Get training schedule
    schedule = scheduler.get_curriculum_schedule()
    
    # Print summary
    scheduler.print_summary(scored_samples)
    
    return scored_samples, phases, schedule


if __name__ == "__main__":
    # Test curriculum scheduler
    print("Testing CurriculumScheduler...")
    
    scheduler = CurriculumScheduler()
    
    # Test difficulty scoring
    test_samples = [
        {
            'question': "Hà Nội ở đâu?",
            'context': "Hà Nội là thủ đô Việt Nam.",
            'answer_start': 0,
            'is_impossible': False
        },
        {
            'question': "Tại sao thành phố này không phải là trung tâm kinh tế lớn nhất dù đã phát triển hơn 1000 năm lịch sử và có nhiều yếu tố thuận lợi?",
            'context': " ".join(["Đây là câu dài. "] * 50),  # Long context
            'answer_start': 800,  # Deep in text
            'is_impossible': True
        },
        {
            'question': "Ai phát triển PhoBERT?",
            'context': "PhoBERT được VinAI phát triển năm 2020.",
            'answer_start': 16,
            'is_impossible': False
        }
    ]
    
    print("\n--- Testing difficulty scoring ---")
    for i, sample in enumerate(test_samples):
        difficulty = scheduler.compute_difficulty(sample)
        print(f"Sample {i+1}: difficulty = {difficulty}/10")
    
    # Test with mock dataset
    print("\n--- Testing dataset scoring ---")
    mock_dataset = test_samples * 10  # Repeat to have more samples
    scored = scheduler.score_dataset(mock_dataset)
    
    scheduler.print_summary(scored)
    
    print("✅ CurriculumScheduler test passed!")
