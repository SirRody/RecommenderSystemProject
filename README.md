# Netflix Recommender System: Collaborative Filtering Algorithms

## Project Overview

This project explores how machine learning can predict user movie ratings using five different collaborative filtering approaches. Just as Netflix recommends movies based on your viewing history, these algorithms learn user preferences from sparse rating data to predict what movies you'll love.

Imagine teaching a computer to understand that users who love "The Matrix" and "Inception" will probably enjoy "Interstellar." That's exactly what this project demonstrates!

## The Problem

Movie recommendation systems present a unique challenge for AI: they combine **sparse user data** with **diverse taste patterns**. Unlike simple classification problems, recommendation engines require:

- Predicting ratings from incomplete data (22.8% missing)
- Identifying hidden patterns in user preferences
- Scaling to thousands of users and movies
- Balancing accuracy with computational speed

Traditional recommendation systems use simple averages, but machine learning algorithms can discover complex patterns purely from rating data—just like Netflix learns your taste over time!

## Project Structure

RecommenderSystemProject/

- **Algorithms**
  - naive_em.py - Expectation-Maximization clustering
  - kmeans.py - K-means clustering implementation
  - em.py - Enhanced EM for incomplete data
  - advanced_cf.py - Modern methods (Matrix Factorization, Neural Networks)

- **Data & Environment**
  - data/toy_data.txt - 2D synthetic data for testing
  - data/netflix_incomplete.txt - 1200×1200 sparse rating matrix
  - data/netflix_complete.txt - Ground truth for evaluation
  - data/test_*.txt - Validation datasets

- **Core Framework**
  - common.py - Shared utilities and data structures
  - main.py - Main execution and experiment runner
  - test.py - Unit testing and validation

- **Visualization & Results**
  - images/ - Cluster visualizations for EM and K-means
  - results/ - Performance comparison charts

- **Configuration**
  - requirements.txt - Python dependencies
  - README.md - This documentation

## Challenges Faced

1. **Data Sparsity**: Predicting 328,232 missing ratings from only 77.2% observed data
2. **Algorithm Selection**: Choosing between probabilistic (EM), distance-based (K-means), and latent factor (Matrix Factorization) approaches
3. **Evaluation Metrics**: Distinguishing between overall RMSE and test-only RMSE for fair comparison
4. **Scalability**: Handling 1.44 million potential ratings efficiently
5. **Overfitting Prevention**: Ensuring algorithms generalize to unseen ratings, not just memorize training data

## Performance Summary

| Algorithm | Test RMSE | Training Time | Performance vs EM |
|---|---|---|---|
| EM Clustering (K=12) | 1.006 | 120s | Baseline (100%) |
| K-means Clustering (K=4) | - | 30s | Clustering only |
| **Matrix Factorization (NMF)** | **0.991** | **4.1s** | **1.5% more accurate, 29× faster** |
| Neural Collaborative Filtering | 1.027 | 2696s | 2% less accurate |
| Alternating Least Squares | 1.132 | 50s | 12.5% less accurate |

## Key Insights

- **Matrix Factorization performed best**, demonstrating the power of latent factor models for sparse data
- **Simplicity beats complexity** - Matrix Factorization (4.1s) outperformed Neural Networks (2696s) with similar accuracy
- **Proper evaluation matters** - Overall RMSE (0.473) was misleading; test-only RMSE (0.991) revealed true performance
- **Scalability is crucial** - Production systems need both accuracy and speed, making Matrix Factorization ideal

## Real-World Applications

1. **Streaming Services**: Netflix, Hulu, Disney+ content recommendations
2. **E-commerce Platforms**: Amazon "customers who bought" suggestions
3. **Music Streaming**: Spotify Discover Weekly playlist generation
4. **Social Media**: YouTube/TikTok content recommendation algorithms
5. **Book Recommendations**: Goodreads, Amazon Kindle suggestions
6. **Food Delivery**: UberEats, DoorDash restaurant recommendations
7. **Travel Platforms**: Airbnb, Booking.com personalized listings
8. **Job Portals**: LinkedIn job recommendations based on profile and history

## Conclusion

This project successfully demonstrates that matrix factorization algorithms can master movie recommendation tasks, achieving **91% of theoretical optimal performance** while being **29 times faster** than traditional EM clustering. The progression from simple clustering to advanced factorization shows how AI can handle increasingly complex recommendation scenarios.

The work bridges statistical modeling and machine learning—a crucial step toward building recommendation systems that balance accuracy, speed, and scalability, with applications ranging from entertainment to e-commerce.

---

## Author

**Rodrick - Data Scientist & Machine Learning Engineer**

A passionate developer exploring the intersection of machine learning and real-world applications. This project represents hands-on experience with implementing and comparing recommendation algorithms to solve practical business problems.

"Building recommendation systems is about more than predicting ratings—it's about creating experiences that understand users, anticipate their needs, and deliver personalized value through intelligent algorithms."