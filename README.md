# Neural Storyteller: Image Captioning with Seq2Seq

An end-to-end deep learning project that generates natural language descriptions for images using a Sequence-to-Sequence (Seq2Seq) architecture. This project was developed as part of the Generative AI (AI4009) course at the National University of Computer and Emerging Sciences.

## 🚀 Features
- **Visual Encoder:** Pre-trained ResNet50 for high-dimensional feature extraction.
- **Language Decoder:** LSTM-based RNN with word embeddings for sequential text generation.
- **Search Algorithms:** Implementation of both **Greedy Search** and **Beam Search**.
- **Interactive App:** Streamlit-based web interface for real-time inference.

## 🛠️ Project Structure
- `AI_ASS01_XXF_YYYY.ipynb`: The complete training and evaluation pipeline.
- `app.py`: Streamlit application code.
- `vocab.pkl`: Serialized vocabulary mapping.
- `flickr30k_model.pth`: Trained model weights (Encoder + Decoder).

## 📊 Performance Metrics
- **Loss:** Successfully converged to ~2.69 Cross-Entropy Loss.
- **BLEU-4 Score:** Evaluated against ground truth captions from the Flickr30k dataset.
- **Evaluation:** Token-level Precision, Recall, and F1-score tracking.
