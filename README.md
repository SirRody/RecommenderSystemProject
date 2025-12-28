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

**Main Code Files (in `src/` folder):**
- `main.py` - Run all experiments
- `common.py` - Shared utilities
- `em.py` - Expectation-Maximization algorithm
- `kmeans.py` - K-means clustering
- `advanced_cf.py` - Matrix Factorization & Neural Networks
- `test.py` - Testing code

**Data Files (in `data/` folder):**
- `toy_data.txt` - Simple test data
- `netflix_incomplete.txt` - Movie ratings with missing values
- `netflix_complete.txt` - All ratings (ground truth)

**Images (in `images/` folder):**
- Cluster visualizations for K-means and EM

**Results (in `results/` folder):**
- Performance comparison charts

**Configuration Files:**
- `requirements.txt` - Python packages needed
- `README.md` - This file
- `.gitignore` - Files to exclude from Git

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

- **Matrix Factorization performed best**, being 1.5% more accurate than EM while 29 times faster
- **Simplicity beats complexity** - Matrix Factorization (4.1s) beat Neural Networks (2696s) despite being simpler
- **Proper evaluation matters** - We learned to use test-only RMSE instead of overall RMSE for fair comparison
- **Scalability is crucial** - Production systems need speed (4 seconds) as much as accuracy (0.99 RMSE)

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

This project demonstrates that matrix factorization achieves **1.5% better accuracy** with **29× faster training** than traditional EM clustering. The real insight: **simpler algorithms often outperform complex ones** when properly implemented.

The work bridges statistical modeling and machine learning—a crucial step toward building recommendation systems that balance accuracy, speed, and scalability, with applications ranging from entertainment to e-commerce.

---

## Author

**Rodrick - Data Scientist & Machine Learning Engineer**

A passionate developer exploring the intersection of machine learning and real-world applications. This project represents hands-on experience with implementing and comparing recommendation algorithms to solve practical business problems.

"Building recommendation systems is about more than predicting ratings—it's about creating experiences that understand users, anticipate their needs, and deliver personalized value through intelligent algorithms."