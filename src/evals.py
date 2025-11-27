import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

@dataclass
class SearchResult:
    """Represents a single search result"""
    image_id: str
    predicted_class: str
    confidence: Optional[float] = None
    actual_class: Optional[str] = None

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    accuracy: float
    precision_per_class: Dict[str, float]
    recall_per_class: Dict[str, float]
    f1_per_class: Dict[str, float]
    confusion_matrix: np.ndarray
    class_names: List[str]

class VisualSearchEvaluator:
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.results: List[SearchResult] = []
        self.ground_truth: Dict[str, str] = {}
        
    def load_ground_truth(self, ground_truth_file: str):
        """Load ground truth labels from file"""
        df = pd.read_csv(ground_truth_file)
        self.ground_truth = dict(zip(df['image_id'], df['true_class']))
        
    def add_search_result(self, image_id: str, predicted_class: str, confidence: float):
        """Add a single search result"""
        actual_class = self.ground_truth.get(image_id)
        result = SearchResult(image_id, predicted_class, confidence, actual_class)
        self.results.append(result)
        
    def add_batch_results(self, predictions: List[Tuple[str, str, float]]):
        """Add multiple search results at once"""
        for image_id, pred_class, confidence in predictions:
            self.add_search_result(image_id, pred_class, confidence)
    
    def calculate_confusion_matrix(self) -> np.ndarray:
        """Calculate confusion matrix"""
        if not self.results:
            raise ValueError("No results to evaluate")
            
        y_true = [r.actual_class for r in self.results if r.actual_class]
        y_pred = [r.predicted_class for r in self.results if r.actual_class]
        
        return confusion_matrix(y_true, y_pred, labels=self.class_names)
    
    def evaluate(self) -> EvaluationMetrics:
        """Perform complete evaluation"""
        y_true = [r.actual_class for r in self.results if r.actual_class]
        y_pred = [r.predicted_class for r in self.results if r.actual_class]
        
        if not y_true:
            raise ValueError("No ground truth data available")
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=self.class_names, average=None
        )
        
        # Convert to dictionaries
        precision_dict = dict(zip(self.class_names, precision))
        recall_dict = dict(zip(self.class_names, recall))
        f1_dict = dict(zip(self.class_names, f1))
        
        cm = self.calculate_confusion_matrix()
        
        return EvaluationMetrics(
            accuracy=accuracy,
            precision_per_class=precision_dict,
            recall_per_class=recall_dict,
            f1_per_class=f1_dict,
            confusion_matrix=cm,
            class_names=self.class_names
        )

# Visualization and Reporting
class EvaluationReporter:
    def __init__(self, evaluator: VisualSearchEvaluator):
        self.evaluator = evaluator
        
    def plot_confusion_matrix(self, metrics: EvaluationMetrics, save_path: Optional[str] = None):
        """Plot and optionally save confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            metrics.confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=metrics.class_names,
            yticklabels=metrics.class_names
        )
        plt.title('Confusion Matrix - Visual Search System')
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_report(self, metrics: EvaluationMetrics) -> str:
        """Generate detailed text report"""
        report = f"""
Visual Search System Evaluation Report
=====================================

Overall Accuracy: {metrics.accuracy:.3f}

Per-Class Performance:
"""
        for class_name in metrics.class_names:
            precision = metrics.precision_per_class[class_name]
            recall = metrics.recall_per_class[class_name]
            f1 = metrics.f1_per_class[class_name]
            
            report += f"""
{class_name}:
  Precision: {precision:.3f}
  Recall:    {recall:.3f}
  F1-Score:  {f1:.3f}
"""
        return report
    
    def save_detailed_results(self, filepath: str):
        """Save detailed results to CSV"""
        results_data = []
        for result in self.evaluator.results:
            results_data.append({
                'image_id': result.image_id,
                'predicted_class': result.predicted_class,
                'actual_class': result.actual_class,
                'confidence': result.confidence,
                'correct': result.predicted_class == result.actual_class
            })
        
        df = pd.DataFrame(results_data)
        df.to_csv(filepath, index=False)

def init_evaluator(base_path: str):
    """Create evaluator/report instances if ground-truth metadata is available."""
    ground_truth_file = os.getenv(
        "GROUND_TRUTH_FILE",
        os.path.join(base_path, "ground_truth_labels.csv")
    )

    if not os.path.exists(ground_truth_file):
        print(f"[Eval] Ground truth file not found at {ground_truth_file}. Skipping evaluation.")
        return None, None

    gt_df = pd.read_csv(ground_truth_file)
    if 'image_id' not in gt_df.columns:
        print(f"[Eval] Missing 'image_id' column in {ground_truth_file}. Skipping evaluation.")
        return None, None

    class_column = 'true_class' if 'true_class' in gt_df.columns else 'class_id'
    if class_column not in gt_df.columns:
        print(f"[Eval] Expected '{class_column}' column not found in {ground_truth_file}. Skipping evaluation.")
        return None, None

    gt_df['true_class'] = gt_df[class_column].astype(str)
    class_names = sorted(gt_df['true_class'].unique(), key=lambda label: int(label))

    evaluator = VisualSearchEvaluator(class_names=class_names)
    evaluator.ground_truth = dict(zip(gt_df['image_id'], gt_df['true_class']))
    reporter = EvaluationReporter(evaluator)
    return evaluator, reporter
