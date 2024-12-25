"""
Complete DiffUse implementation for text classification task
Using transformers for embeddings and scikit-learn for classification
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModel
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

@dataclass
class EmbeddingPair:
    """Container for a pair of embeddings and their metadata"""
    embedding_a: np.ndarray
    embedding_b: np.ndarray
    metadata: Dict[str, Any]
    label: Optional[int] = None

class TransformerEmbeddingModel:
    """Wrapper for transformer-based embedding models"""
    
    def __init__(self, model_name: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device
        
    def get_embedding(self, text: str) -> np.ndarray:
        # Tokenize and move to device
        inputs = self.tokenizer(text, 
                              padding=True, 
                              truncation=True, 
                              max_length=512, 
                              return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings[0]  # Return single embedding

class EmbeddingGenerator:
    """Generates embeddings using two different transformer models"""
    
    def __init__(self, model_a_name: str, model_b_name: str):
        self.model_a = TransformerEmbeddingModel(model_a_name)
        self.model_b = TransformerEmbeddingModel(model_b_name)

    def generate_embeddings(self, texts: List[str], labels: Optional[List[int]] = None) -> List[EmbeddingPair]:
        embedding_pairs = []
        
        for idx, text in enumerate(tqdm(texts, desc="Generating embeddings")):
            emb_a = self.model_a.get_embedding(text)
            emb_b = self.model_b.get_embedding(text)
            
            label = labels[idx] if labels is not None else None
            
            pair = EmbeddingPair(
                embedding_a=emb_a,
                embedding_b=emb_b,
                metadata={"text": text},
                label=label
            )
            embedding_pairs.append(pair)
            
        return embedding_pairs

class DifferenceVectorGenerator:
    """Computes difference vectors between embedding pairs"""
    
    def generate_difference_vectors(self, embedding_pairs: List[EmbeddingPair]) -> np.ndarray:
        diff_vectors = []
        for pair in embedding_pairs:
            # Ensure embeddings are numpy arrays and perform subtraction
            emb_a = np.asarray(pair.embedding_a)
            emb_b = np.asarray(pair.embedding_b)
            diff = emb_a - emb_b
            diff_vectors.append(diff)
        return np.array(diff_vectors)

class ClusterAnalyzer:
    """Clusters difference vectors and selects representatives ensuring class balance"""
    
    def __init__(self, n_clusters: int, min_samples_per_class: int = 3):
        self.n_clusters = n_clusters
        self.min_samples_per_class = min_samples_per_class
        self.clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'
        )

    def find_representatives(self, diff_vectors: np.ndarray, labels: np.ndarray) -> List[int]:
        # Perform clustering
        cluster_labels = self.clustering.fit_predict(diff_vectors)
        
        # Get unique classes
        unique_classes = np.unique(labels)
        if len(unique_classes) < 2:
            raise ValueError("Need at least two classes for classification")
            
        # Select representatives ensuring class balance and minimum samples
        representatives = []
        
        for class_label in unique_classes:
            # Get indices for this class
            class_indices = np.where(labels == class_label)[0]
            
            # Get clusters that contain examples of this class
            class_clusters = np.unique(cluster_labels[class_indices])
            
            # Track selected samples for this class
            class_representatives = []
            
            # Try to get samples from different clusters first
            for cluster_id in class_clusters:
                cluster_class_indices = class_indices[
                    cluster_labels[class_indices] == cluster_id
                ]
                
                if len(cluster_class_indices) > 0:
                    representative = self._select_representative(
                        diff_vectors[cluster_class_indices],
                        cluster_class_indices
                    )
                    if representative not in representatives:
                        class_representatives.append(representative)
            
            # If we still need more samples, add additional ones from the largest clusters
            while len(class_representatives) < self.min_samples_per_class and len(class_indices) >= self.min_samples_per_class:
                # Sort clusters by size
                cluster_sizes = [(c, np.sum(cluster_labels[class_indices] == c)) 
                               for c in class_clusters]
                cluster_sizes.sort(key=lambda x: x[1], reverse=True)
                
                for cluster_id, _ in cluster_sizes:
                    cluster_class_indices = class_indices[
                        cluster_labels[class_indices] == cluster_id
                    ]
                    # Remove already selected indices
                    cluster_class_indices = [i for i in cluster_class_indices 
                                          if i not in class_representatives]
                    
                    if len(cluster_class_indices) > 0:
                        representative = self._select_representative(
                            diff_vectors[cluster_class_indices],
                            cluster_class_indices
                        )
                        if representative not in class_representatives:
                            class_representatives.append(representative)
                            break
                
                # Break if we can't find any more unique representatives
                if len(class_representatives) == len(class_indices):
                    break
            
            if len(class_representatives) < self.min_samples_per_class:
                raise ValueError(f"Could not find {self.min_samples_per_class} representatives for class {class_label}. Got {len(class_representatives)} samples.")
            
            representatives.extend(class_representatives)
                        
        return representatives

    def _select_representative(self, cluster_vectors: np.ndarray, original_indices: np.ndarray) -> int:
        # Select example closest to cluster center
        center = np.mean(cluster_vectors, axis=0)
        distances = np.linalg.norm(cluster_vectors - center, axis=1)
        closest_idx = np.argmin(distances)
        return original_indices[closest_idx]

class ClassificationEvaluator:
    """Evaluates embeddings for text classification task"""
    
    def __init__(self, classifier_model: Any):
        self.classifier = classifier_model

    def evaluate(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        # Ensure we have at least 2 samples per class
        unique_classes = np.unique(labels)
        if len(unique_classes) < 2:
            raise ValueError("Need at least two classes for evaluation")

        # Check if we have enough samples for a meaningful split
        min_train_samples = 2  # minimum samples needed per class for training
        min_test_samples = 1   # minimum samples needed per class for testing
        
        for class_label in unique_classes:
            class_samples = np.sum(labels == class_label)
            if class_samples < (min_train_samples + min_test_samples):
                raise ValueError(f"Need at least {min_train_samples + min_test_samples} samples per class. Class {class_label} has only {class_samples} samples.")

        # Split ensuring class representation in both sets
        train_indices = []
        test_indices = []
        
        for class_label in unique_classes:
            class_indices = np.where(labels == class_label)[0]
            np.random.shuffle(class_indices)
            
            # Take minimum required samples for training
            train_indices.extend(class_indices[:min_train_samples])
            # Take remaining samples for testing
            test_indices.extend(class_indices[min_train_samples:])

        # Train classifier
        self.classifier.fit(embeddings[train_indices], labels[train_indices])
        
        # Predict on test set
        predictions = self.classifier.predict(embeddings[test_indices])
        true_labels = labels[test_indices]
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(true_labels, predictions),
            "f1_score": f1_score(true_labels, predictions, average='weighted'),
            "train_samples": len(train_indices),
            "test_samples": len(test_indices)
        }
        
        return metrics

class EmbeddingSelector:
    """Coordinates the overall embedding selection process"""
    
    def __init__(self,
                 embedding_generator: EmbeddingGenerator,
                 diff_generator: DifferenceVectorGenerator,
                 cluster_analyzer: ClusterAnalyzer,
                 evaluator: ClassificationEvaluator):
        self.embedding_generator = embedding_generator
        self.diff_generator = diff_generator
        self.cluster_analyzer = cluster_analyzer
        self.evaluator = evaluator

    def select_embeddings(self,
                         texts: List[str],
                         labels: List[int]) -> Tuple[List[int], Dict[str, float]]:
        # Generate embeddings
        print("Step 1: Generating embeddings...")
        embedding_pairs = self.embedding_generator.generate_embeddings(texts, labels)
        
        # Generate difference vectors
        print("Step 2: Computing difference vectors...")
        diff_vectors = self.diff_generator.generate_difference_vectors(embedding_pairs)
        
        # Find representatives through clustering
        print("Step 3: Clustering and selecting representatives...")
        labels_array = np.array([pair.label for pair in embedding_pairs])
        selected_indices = self.cluster_analyzer.find_representatives(diff_vectors, labels_array)
        
        # Evaluate selected embeddings
        print("Step 4: Evaluating selected embeddings...")
        selected_embeddings = np.array([embedding_pairs[i].embedding_a for i in selected_indices])
        selected_labels = np.array([embedding_pairs[i].label for i in selected_indices])
        
        metrics = self.evaluator.evaluate(selected_embeddings, selected_labels)
        
        return selected_indices, metrics

# Example usage with text classification:
def main():
    from sklearn.svm import SVC
    
    # Sample data with more examples per class
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
    # 1: positive, 0: negative - 8 samples per class
    labels = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    
    # Initialize components
    embedding_gen = EmbeddingGenerator(
        model_a_name='bert-base-uncased',
        model_b_name='roberta-base'
    )
    
    diff_gen = DifferenceVectorGenerator()
    cluster_analyzer = ClusterAnalyzer(n_clusters=4, min_samples_per_class=3)
    evaluator = ClassificationEvaluator(classifier_model=SVC(kernel='rbf', probability=True))
    
    # Create selector
    selector = EmbeddingSelector(
        embedding_generator=embedding_gen,
        diff_generator=diff_gen,
        cluster_analyzer=cluster_analyzer,
        evaluator=evaluator
    )
    
    # Run selection process
    try:
        selected_indices, metrics = selector.select_embeddings(texts, labels)
        
        print("\nResults:")
        print(f"Selected indices: {selected_indices}")
        print(f"Evaluation metrics:")
        for metric, value in metrics.items():
            print(f"- {metric}: {value:.3f}" if isinstance(value, float) else f"- {metric}: {value}")
        
        print("\nSelected texts:")
        for idx in selected_indices:
            print(f"- {texts[idx]} (Label: {labels[idx]})")
            
    except ValueError as e:
        print(f"Error: {e}")
        print("Try increasing the number of samples per class or adjusting the clustering parameters.")

if __name__ == "__main__":
    main()