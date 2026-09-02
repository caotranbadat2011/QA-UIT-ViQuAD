"""
Advanced metrics and error analysis for Vietnamese QA system.
Provides detailed evaluation beyond basic EM/F1 scores.
"""
import json
import numpy as np
from typing import Dict, List, Tuple
from collections import Counter


class ErrorAnalyzer:
    """Analyze prediction errors in detail."""
    
    def __init__(self, predictions: Dict, ground_truth: List[Dict]):
        """
        Args:
            predictions: Dict mapping example_id -> predicted_answer
            ground_truth: List of dicts with id, question, context, answers
        """
        self.predictions = predictions
        self.ground_truth = {ex['id']: ex for ex in ground_truth}
        
    def categorize_error(self, gold: str, pred: str) -> str:
        """Categorize the type of error."""
        if not gold and not pred:
            return "correct_no_answer"
        if not gold and pred:
            return "false_positive"
        if gold and not pred:
            return "false_negative"
        
        # Both non-empty
        gold_norm = self._normalize(gold)
        pred_norm = self._normalize(pred)
        
        if gold_norm == pred_norm:
            return "correct"
        
        # Analyze error type
        gold_words = set(gold_norm.split())
        pred_words = set(pred_norm.split())
        
        overlap = len(gold_words & pred_words)
        union = len(gold_words | pred_words)
        
        if overlap == 0:
            return "completely_wrong"
        
        iou = overlap / union if union > 0 else 0
        
        if iou > 0.5:
            return "partial_overlap"
        elif len(pred_words) > len(gold_words) * 2:
            return "too_long"
        elif len(pred_words) < len(gold_words) * 0.5:
            return "too_short"
        else:
            return "different_span"
    
    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        return ' '.join(text.lower().strip().split())
    
    def build_confusion_matrix(self) -> Dict:
        """
        Build confusion matrix for HasAns vs NoAns classification.
        
        Returns:
            Dict with tp, tn, fp, fn counts
        """
        tp = tn = fp = fn = 0
        
        for ex_id, gold_data in self.ground_truth.items():
            pred = self.predictions.get(ex_id, "")
            has_gold = len(gold_data.get('answers', {}).get('text', [])) > 0
            has_pred = bool(pred and pred.strip())
            
            if has_gold and has_pred:
                tp += 1
            elif not has_gold and not has_pred:
                tn += 1
            elif has_gold and not has_pred:
                fn += 1
            else:  # not has_gold and has_pred
                fp += 1
        
        return {
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'matrix': [[tp, fn], [fp, tn]]
        }
    
    def analyze_by_category(self) -> Dict:
        """Analyze performance by error category."""
        categories = Counter()
        category_correct = Counter()
        
        for ex_id, gold_data in self.ground_truth.items():
            gold_answers = gold_data.get('answers', {}).get('text', [])
            gold_text = gold_answers[0] if gold_answers else ""
            pred = self.predictions.get(ex_id, "")
            
            error_type = self.categorize_error(gold_text, pred)
            categories[error_type] += 1
            
            if error_type == "correct" or error_type == "correct_no_answer":
                category_correct[error_type] += 1
        
        # Calculate accuracy per category
        results = {}
        for cat, count in categories.items():
            correct = category_correct.get(cat, 0)
            results[cat] = {
                'count': count,
                'correct': correct,
                'accuracy': correct / count if count > 0 else 0
            }
        
        return results
    
    def generate_bad_cases_report(self, n: int = 20) -> List[Dict]:
        """
        Generate report of worst predictions.
        
        Args:
            n: Number of bad cases to return
            
        Returns:
            List of dicts with error details
        """
        bad_cases = []
        
        for ex_id, gold_data in self.ground_truth.items():
            gold_answers = gold_data.get('answers', {}).get('text', [])
            gold_text = gold_answers[0] if gold_answers else ""
            pred = self.predictions.get(ex_id, "")
            
            error_type = self.categorize_error(gold_text, pred)
            
            if error_type not in ["correct", "correct_no_answer"]:
                bad_cases.append({
                    'id': ex_id,
                    'question': gold_data.get('question', ''),
                    'context': gold_data.get('context', '')[:200] + '...',
                    'gold_answer': gold_text,
                    'predicted_answer': pred,
                    'error_type': error_type
                })
        
        # Sort by severity (completely_wrong first)
        severity_order = {
            'completely_wrong': 0,
            'false_positive': 1,
            'false_negative': 2,
            'too_long': 3,
            'too_short': 4,
            'different_span': 5,
            'partial_overlap': 6
        }
        
        bad_cases.sort(key=lambda x: severity_order.get(x['error_type'], 99))
        
        return bad_cases[:n]
    
    def compute_detailed_metrics(self) -> Dict:
        """Compute comprehensive evaluation metrics."""
        try:
            from evaluate import compute_exact, compute_f1
        except ImportError:  # goi theo kieu package: python -m src.metrics_advanced
            from src.evaluate import compute_exact, compute_f1
        
        em_scores = []
        f1_scores = []
        has_ans_em = []
        has_ans_f1 = []
        no_ans_correct = 0
        no_ans_total = 0
        
        for ex_id, gold_data in self.ground_truth.items():
            gold_answers = gold_data.get('answers', {}).get('text', [])
            pred = self.predictions.get(ex_id, "")
            
            is_impossible = len(gold_answers) == 0
            
            if is_impossible:
                no_ans_total += 1
                if pred == "":
                    no_ans_correct += 1
                em = 1 if pred == "" else 0
                f1 = em
            else:
                em = max(compute_exact(g, pred) for g in gold_answers)
                f1 = max(compute_f1(g, pred) for g in gold_answers)
                has_ans_em.append(em)
                has_ans_f1.append(f1)
            
            em_scores.append(em)
            f1_scores.append(f1)
        
        metrics = {
            'overall_em': float(np.mean(em_scores)) * 100 if em_scores else 0,
            'overall_f1': float(np.mean(f1_scores)) * 100 if f1_scores else 0,
            'has_ans_em': float(np.mean(has_ans_em)) * 100 if has_ans_em else 0,
            'has_ans_f1': float(np.mean(has_ans_f1)) * 100 if has_ans_f1 else 0,
            'no_ans_accuracy': (no_ans_correct / no_ans_total * 100) if no_ans_total > 0 else 0,
            'total_samples': len(self.ground_truth),
            'answerable_count': len(has_ans_em),
            'unanswerable_count': no_ans_total
        }
        
        return metrics


def run_full_analysis(predictions_file: str, test_data_file: str, 
                      output_file: str = "error_analysis.json"):
    """
    Run complete error analysis and save results.
    
    Args:
        predictions_file: Path to predictions.json
        test_data_file: Path to test dataset (parquet or json)
        output_file: Output file for analysis results
    """
    import pandas as pd
    from datasets import Dataset
    
    # Load predictions
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    # Load test data
    if test_data_file.endswith('.parquet'):
        df = pd.read_parquet(test_data_file)
        
        def format_answers(row):
            if row.get("is_impossible", False) or row.get("answer_start", -1) == -1:
                return {"text": [], "answer_start": []}
            return {
                "text": [str(row["answer_text"])], 
                "answer_start": [int(row["answer_start"])]
            }
        
        df["answers"] = df.apply(format_answers, axis=1)
        df["id"] = df["qa_id"]
        df["question"] = df["question"].astype(str).str.strip()
        
        test_ds = Dataset.from_pandas(df[["id", "question", "context", "answers"]])
    else:
        with open(test_data_file, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        rows = []
        for article in raw["data"]:
            for para in article["paragraphs"]:
                context = para["context"]
                for qa in para["qas"]:
                    answers = qa.get("answers", [])
                    rows.append({
                        "id": qa["id"],
                        "question": qa["question"].strip(),
                        "context": context,
                        "answers": {
                            "text": [a["text"] for a in answers],
                            "answer_start": [a["answer_start"] for a in answers],
                        },
                    })
        test_ds = Dataset.from_list(rows)
    
    # Run analysis
    analyzer = ErrorAnalyzer(predictions, list(test_ds))
    
    results = {
        'detailed_metrics': analyzer.compute_detailed_metrics(),
        'confusion_matrix': analyzer.build_confusion_matrix(),
        'error_categories': analyzer.analyze_by_category(),
        'bad_cases': analyzer.generate_bad_cases_report(n=20)
    }
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Analysis saved to {output_file}")
    print(f"\nOverall EM: {results['detailed_metrics']['overall_em']:.2f}%")
    print(f"Overall F1: {results['detailed_metrics']['overall_f1']:.2f}%")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run advanced error analysis")
    parser.add_argument("--predictions", default="predictions.json")
    parser.add_argument("--test_file", default="data/processed/test.parquet")
    parser.add_argument("--output", default="error_analysis.json")
    
    args = parser.parse_args()
    
    run_full_analysis(args.predictions, args.test_file, args.output)
