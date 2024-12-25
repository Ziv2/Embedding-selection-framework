
# Embedding Model Selection Framework

A comprehensive framework for evaluating and selecting the most suitable embedding model for various NLP tasks. This tool provides detailed performance metrics, cost analysis, and resource utilization comparisons across different embedding providers including OpenAI, VoyageAI, Hugging Face models, and Sentence Transformers.

## 🌟 Features

- Comprehensive embedding model evaluation framework
- Support for multiple embedding providers:
  - OpenAI (text-embedding-3-small, text-embedding-3-large)
  - VoyageAI (voyage-2, voyage-3)
  - Hugging Face models (BERT, DistilBERT)
  - Sentence Transformers
- Detailed performance metrics and resource monitoring
- Cost analysis and estimation
- Task-specific model recommendations
- Automated model selection based on requirements

## 📋 Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, but recommended for better performance)
- API keys for OpenAI and VoyageAI (if using these providers)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/embedding-selection-framework.git
cd embedding-selection-framework
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables for API keys:
```bash
export OPENAI_API_KEY='your-openai-api-key'
export VOYAGE_API_KEY='your-voyage-api-key'
```

## 💻 Usage

### Basic Usage

```python
from embedding_evaluator import EnhancedEmbeddingEvaluator
from embedding_providers import OpenAIEmbedding, VoyageAIEmbedding, HuggingFaceEmbedding

# Initialize providers
providers = [
    HuggingFaceEmbedding('bert-base-uncased'),
    OpenAIEmbedding('text-embedding-3-small'),
    VoyageAIEmbedding('voyage-2')
]

# Compare embeddings
results = EnhancedEmbeddingEvaluator.compare_embeddings(
    embedding_providers=providers,
    classifier_factory=lambda: SVC(kernel='linear'),
    train_texts=train_texts,
    train_labels=train_labels,
    test_texts=test_texts,
    test_labels=test_labels
)
```

### Task-Specific Model Selection

```python
from nlp_embeddings import NLPEmbeddingHandler

handler = NLPEmbeddingHandler()
result = handler.get_embeddings('sentiment_analysis', "This product is amazing!")
```

## 📊 Performance Metrics

The framework provides comprehensive metrics including:
- Accuracy, Precision, Recall, F1 score
- Processing speed (tokens/second)
- Memory usage (RAM and GPU)
- Cost estimates
- Embedding dimensions
- Total processing time

## 🛠️ Configuration

The framework supports various configuration options through environment variables or a config file:

```python
MODEL_SPECS = {
    'OpenAI_text-embedding-3-small': {
        'dim': 1536,
        'cost_per_1k_tokens': 0.00002,
        'max_tokens': 8191,
        'suggested_batch': 2048
    },
    # ... more model specifications
}
```

## 📈 Example Results

```
Model                         Accuracy    F1-Score    Tokens/sec    Cost/1M tokens
---------------------------- ----------- ----------- ------------- ----------------
OpenAI_text-embedding-3-small    0.945      0.943        5000         $0.02
VoyageAI_voyage-2               0.932      0.930        4500         $0.12
HF_bert-base-uncased           0.918      0.915        2000         $0.00
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face for their transformer models
- Sentence Transformers team
- OpenAI for their embedding models and API
- VoyageAI for their embedding services

## 📞 Contact

https://www.deepkeep.ai/contact
