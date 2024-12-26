"""
Benchmark script for comparing embedding models using DiffUse methodology.
Tracks performance metrics, resource usage, and costs.
"""

import time
import psutil
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from dataclasses import dataclass
import gc
from transformers import AutoTokenizer, AutoModel
import GPUtil
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
from pathlib import Path

@dataclass
class ModelMetrics:
    """Container for model performance metrics"""
    model_name: str
    total_time_seconds: float
    peak_memory_mb: float
    gpu_memory_mb: float
    avg_cpu_percent: float
    total_tokens: int
    avg_tokens_per_second: float
    total_cost_usd: float
    embedding_dimension: int
    max_sequence_length: int
    suggested_batch_size: int
    total_embedding_size_mb: float
    accuracy: float
    precision: float
    recall: float
    f1: float

class ModelBenchmark:
    """Benchmark different embedding models"""
    
    # Cost estimates per 1000 tokens (adjust as needed)
    COST_PER_1K_TOKENS = {
        'bert-base-uncased': 0.0001,
        'roberta-base': 0.0001,
        'albert-base-v2': 0.0001,
        'distilbert-base-uncased': 0.00005,
        'microsoft/mpnet-base': 0.0002,
        'sentence-transformers/all-MiniLM-L6-v2': 0.00005
    }

    def __init__(self, 
                 model_names: List[str],
                 texts: List[str],
                 labels: List[int],
                 batch_size: int = 32,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model_names = model_names
        self.texts = texts
        self.labels = labels
        self.batch_size = batch_size
        self.device = device
        self.metrics = {}

    def _get_gpu_memory(self) -> float:
        """Get GPU memory usage in MB"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].memoryUsed
            return 0.0
        except:
            return 0.0

    def _estimate_cost(self, model_name: str, total_tokens: int) -> float:
        """Estimate cost based on token usage"""
        base_name = model_name.split('/')[-1]
        cost_per_1k = self.COST_PER_1K_TOKENS.get(base_name, 0.0001)
        return (total_tokens / 1000) * cost_per_1k

    def _get_model_info(self, model: AutoModel, tokenizer: AutoTokenizer) -> Dict[str, int]:
        """Get model architecture information"""
        info = {
            'embedding_dimension': model.config.hidden_size,
            'max_sequence_length': tokenizer.model_max_length,
            'suggested_batch_size': self.batch_size
        }
        # Estimate total embedding size in MB
        params_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        info['total_embedding_size_mb'] = params_size
        return info

    def benchmark_model(self, model_name: str) -> ModelMetrics:
        """Run comprehensive benchmark for a single model"""
        print(f"\nBenchmarking {model_name}...")
        
        # Initialize metrics tracking
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        cpu_percentages = []
        total_tokens = 0
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(self.device)
        model.eval()

        embeddings = []
        
        # Process texts in batches
        for i in range(0, len(self.texts), self.batch_size):
            batch_texts = self.texts[i:i + self.batch_size]
            
            # Track CPU usage
            cpu_percentages.append(psutil.Process().cpu_percent())
            
            # Tokenize and get token count
            inputs = tokenizer(batch_texts, 
                             padding=True,
                             truncation=True,
                             return_tensors="pt")
            total_tokens += inputs['input_ids'].numel()
            
            # Generate embeddings
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(batch_embeddings)

        embeddings = np.array(embeddings)
        
        # Calculate performance metrics
        accuracy, precision, recall, f1 = self._calculate_metrics(embeddings)
        
        # Gather all metrics
        end_time = time.time()
        total_time = end_time - start_time
        
        metrics = ModelMetrics(
            model_name=model_name,
            total_time_seconds=total_time,
            peak_memory_mb=psutil.Process().memory_info().rss / (1024 * 1024) - initial_memory,
            gpu_memory_mb=self._get_gpu_memory(),
            avg_cpu_percent=np.mean(cpu_percentages),
            total_tokens=total_tokens,
            avg_tokens_per_second=total_tokens / total_time,
            total_cost_usd=self._estimate_cost(model_name, total_tokens),
            **self._get_model_info(model, tokenizer),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1
        )
        
        # Cleanup
        del model, tokenizer, embeddings
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        return metrics

    def _calculate_metrics(self, embeddings: np.ndarray) -> tuple:
        """Calculate classification metrics using embeddings"""
        # Simple classifier for evaluation
        from sklearn.svm import SVC
        from sklearn.model_selection import train_test_split
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, self.labels, test_size=0.2, random_state=42
        )
        
        # Train and evaluate
        clf = SVC(kernel='rbf')
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, average='weighted'
        )
        
        return accuracy, precision, recall, f1

    def run_benchmarks(self) -> pd.DataFrame:
        """Run benchmarks for all models"""
        results = []
        
        for model_name in self.model_names:
            try:
                metrics = self.benchmark_model(model_name)
                results.append(vars(metrics))
            except Exception as e:
                print(f"Error benchmarking {model_name}: {str(e)}")
                
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        self._save_results(df)
        
        return df
    
    def _save_results(self, df: pd.DataFrame):
        """Save benchmark results"""
        # Create results directory
        results_dir = Path('benchmark_results')
        results_dir.mkdir(exist_ok=True)
        
        # Save CSV
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        df.to_csv(results_dir / f'benchmark_results_{timestamp}.csv', index=False)
        
        # Save detailed JSON with model configs
        detailed_results = df.to_dict('records')
        with open(results_dir / f'benchmark_details_{timestamp}.json', 'w') as f:
            json.dump(detailed_results, f, indent=2)

def main():
    # Example usage
    model_names = [
        'bert-base-uncased',
        'roberta-base',
        'albert-base-v2',
        'distilbert-base-uncased',
        'microsoft/mpnet-base',
        'sentence-transformers/all-MiniLM-L6-v2'
    ]
    
    # Sample texts and labels (extend as needed)
    texts = [
        # Positive reviews
        "This movie is great and entertaining",
        "An amazing performance by the actors",
        "Highly recommended movie",
        "Best film I've ever seen",
        "Brilliant plot and execution",
        "Outstanding cinematography and direction",
        "A masterpiece of modern cinema",
        "Wonderful storytelling and acting",
        
        # Negative reviews
        "I really disliked this film",
        "Terrible waste of time",
        "The worst movie ever made",
        "Poor acting and boring story",
        "Complete disappointment",
        "Awful screenplay and direction",
        "A total disaster of a movie",
        "Dreadful experience, avoid at all costs"
    ]
    labels = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    
    # Run benchmarks
    benchmark = ModelBenchmark(model_names, texts, labels)
    results_df = benchmark.run_benchmarks()
    
    # Print results
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print("\nBenchmark Results:")
    print(results_df)

if __name__ == "__main__":
    main()
