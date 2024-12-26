from typing import Union, List, Tuple, Dict
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

class NLPEmbeddingHandler:
    def __init__(self):
        self.task_to_model = {
            'classification': 'bert-base-uncased',
            'ner': 'bert-base-cased',
            'intent_recognition': 'facebook/bart-large',
            'link_prediction': 'sentence-transformers/all-MiniLM-L6-v2',
            'anomaly_detection': 'roberta-base',
            'fake_news_detection': 'distilbert-base-uncased',
            'relation_extraction': 'xlnet-base-cased',
            'sentiment_analysis': 'distilbert-base-uncased',
            'recommendation': 'sentence-transformers/all-MiniLM-L6-v2',
            'clustering': 'sentence-transformers/all-mpnet-base-v2',
            'topic_modeling': 'gpt2',
            'semantic_search': 'sentence-transformers/msmarco-distilbert-base-v4'
        }
        
    def identify_input_type(self, text_input: Union[str, List[str]]) -> Tuple[str, tuple]:
        """
        Identifies the type and shape of the input text.
        
        Args:
            text_input: Input text that could be a word, list of words, sentence, or list of sentences
            
        Returns:
            Tuple containing input_type and shape
        """
        if isinstance(text_input, str):
            words = text_input.split()
            if len(words) == 1:
                return 'word', (1,)
            elif len(words) <= 50:  # Arbitrary threshold for sentence vs. long text
                return 'sentence', (len(words),)
            else:
                return 'long_text', (len(words),)
        
        elif isinstance(text_input, list):
            if not text_input:
                raise ValueError("Empty list provided")
            
            # Check if it's a list of words or sentences
            max_words = max(len(str(item).split()) for item in text_input)
            if max_words == 1:
                return 'word_list', (len(text_input),)
            else:
                return 'sentence_list', (len(text_input), max_words)
        
        else:
            raise ValueError("Input must be either string or list of strings")

    def get_embeddings(self, 
                      task_type: str, 
                      text_input: Union[str, List[str]]) -> Dict:
        """
        Get embeddings based on task type and input text.
        
        Args:
            task_type: Type of NLP task (classification, ner, etc.)
            text_input: Input text to embed
            
        Returns:
            Dictionary containing embeddings and metadata
        """
        # Normalize task type
        task_type = task_type.lower().replace(' ', '_')
        
        # Validate task type
        if task_type not in self.task_to_model:
            raise ValueError(f"Unsupported task type. Supported types: {list(self.task_to_model.keys())}")
        
        # Get input type and shape
        input_type, shape = self.identify_input_type(text_input)
        
        # Load appropriate model and tokenizer
        model_name = self.task_to_model[task_type]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Tokenize and get embeddings
        with torch.no_grad():
            inputs = tokenizer(text_input, 
                             padding=True, 
                             truncation=True, 
                             return_tensors="pt",
                             max_length=512)
            
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        
        return {
            'task_type': task_type,
            'input_type': input_type,
            'input_shape': shape,
            'model_used': model_name,
            'embedding_shape': embeddings.shape,
            'embeddings': embeddings
        }

# Example usage
if __name__ == "__main__":
    handler = NLPEmbeddingHandler()
    
    # Example 1: Single word classification
    text1 = "hello"
    result1 = handler.get_embeddings('classification', text1)
    print("\nExample 1 - Single word:")
    print(f"Input type: {result1['input_type']}")
    print(f"Input shape: {result1['input_shape']}")
    print(f"Embedding shape: {result1['embedding_shape']}")
    
    # Example 2: Sentence for sentiment analysis
    text2 = "This movie was absolutely fantastic!"
    result2 = handler.get_embeddings('sentiment_analysis', text2)
    print("\nExample 2 - Sentence:")
    print(f"Input type: {result2['input_type']}")
    print(f"Input shape: {result2['input_shape']}")
    print(f"Embedding shape: {result2['embedding_shape']}")
    
    # Example 3: List of sentences for clustering
    text3 = ["First sentence here.", "Another different sentence.", "Third unique sentence."]
    result3 = handler.get_embeddings('clustering', text3)
    print("\nExample 3 - List of sentences:")
    print(f"Input type: {result3['input_type']}")
    print(f"Input shape: {result3['input_shape']}")
    print(f"Embedding shape: {result3['embedding_shape']}")
