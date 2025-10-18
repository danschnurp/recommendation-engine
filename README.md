# Product Recommendation System
 
A Case Study on E-commerce Data for recommendation system for e-commerce using multiple algorithms: SVD Matrix Factorization, Association Rules (Apriori), and Item2Vec (Word2Vec for products).

## Dataset

**Online Retail Dataset** from UCI Machine Learning Repository
- ~540,000 transactions
- ~4,000 products
- ~4,000 customers
- Period: 2010-2011

Download: https://archive.ics.uci.edu/ml/datasets/online+retail

## Installation
 - Install the required packages using the following command:
   - poetry install


## Models Overview

### SVD (Singular Value Decomposition)
- **Type:** Matrix Factorization
- **Use case:** Personalized recommendations
- **Strengths:** Handles sparse data, scalable
- **Output:** Top-N products per user

### Apriori (Association Rules)
- **Type:** Market Basket Analysis
- **Use case:** "Frequently bought together"
- **Strengths:** Interpretable, discovers product associations
- **Output:** Rules (IF bought X THEN buy Y)

### Item2Vec (Word2Vec for Products)
- **Type:** Neural Embedding
- **Use case:** Product similarity, semantic relationships
- **Strengths:** Captures complex patterns, clustering
- **Output:** Product embeddings + similarities

## Evaluation Metrics

All models are evaluated using:
- **Precision@K**: Accuracy of top-K recommendations
- **Recall@K**: Coverage of user's actual purchases
- **F1@K**: Harmonic mean of Precision and Recall



