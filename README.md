# 🎬 Netflix-Style Movie Recommender System
### Building Smarter Movie Recommendations with Machine Learning

## 📊 Project Overview
Have you ever wondered how Netflix knows exactly what movies you'll love? This project builds the same kind of smart recommendation system that streaming services use. We compare 5 different algorithms to find the best way to predict movie ratings.

**What we achieved:**
- ✅ **0.47 star average error** - Almost like knowing what you'd rate a movie!
- ✅ **5 algorithms tested** - From basic to advanced machine learning
- ✅ **Real data used** - 1,200 users, 1,200 movies, 1.4 million potential ratings
- ✅ **Production-ready results** - Found the best balance of speed and accuracy

## 🎯 The Problem We Solved
**Imagine this:** Netflix has millions of users and thousands of movies. Most users only rate a few movies. How can Netflix guess what rating you'd give to a movie you haven't seen yet?

**Our challenge:** 
- We had data for 1,200 users and 1,200 movies
- 22.8% of ratings were missing (328,232 predictions needed!)
- We could only use the existing ratings to guess the missing ones

## 🔬 The Algorithms We Tested

### **1. EM Clustering (The Classic)**
- **How it works:** Groups similar users together
- **Like:** Putting people in "horror movie lovers" or "rom-com fans" groups
- **Result:** Good baseline, but slower

### **2. K-means (The Simple)**
- **How it works:** Strict grouping of users
- **Like:** Sorting people into clear boxes
- **Result:** Fast, but less accurate for complex tastes

### **3. Matrix Factorization (The Winner!)**
- **How it works:** Finds hidden patterns in user preferences
- **Like:** Discovering that people who like Sci-Fi also tend to like certain directors
- **Result:** **Best overall** - fast and accurate

### **4. Neural Network (The Fancy)**
- **How it works:** Complex AI that learns deep patterns
- **Like:** Having a super-smart assistant that learns your taste
- **Result:** Powerful but slow

### **5. Alternating Least Squares (The Industrial)**
- **How it works:** Optimized for large systems
- **Like:** How big companies handle millions of users
- **Result:** Good for scaling up

## 📊 Results: Who Won?

| Method | Accuracy (Error) | Speed | Best For |
|--------|------------------|-------|----------|
| EM Clustering | 1.0 stars error | 2 minutes | Learning the basics |
| **Matrix Factorization** | **0.47 stars error** | **4 seconds** | **Real-world use** |
| Neural Network | 0.49 stars error | 45 minutes | Complex patterns |
| K-means | - | Fast | Simple grouping |

**Key Finding:** Matrix Factorization was the **clear winner** - almost as accurate as the fanciest methods but 600 times faster!

## 🚀 How to Use This Project

### **Installation (3 Steps):**
1. **Open Command Prompt**
2. **Type:** `pip install numpy scikit-learn matplotlib`
3. **Press Enter** (wait for installation)

### **Running the Project:**
1. **Navigate to project folder:**