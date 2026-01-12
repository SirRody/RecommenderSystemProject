# 📊 Sentiment Analysis: Amazon Food Reviews Classifier

## 🎯 Project Overview
This project builds a **machine learning system** that automatically reads Amazon food reviews and determines if they're positive or negative. Think of it as a **robot taste-tester** that learns from thousands of customer opinions!

**Key Stats:**
- 3 machine learning algorithms implemented from scratch
- 13,108 unique words analyzed
- 80.8% accuracy on unseen reviews
- 5,000+ lines of training data

## 🤔 The Problem
Online shoppers face **review overload** - thousands of reviews for each product. How do you quickly understand what most customers think? This project solves that by automatically classifying reviews as **👍 Positive** or **👎 Negative**, helping shoppers make faster decisions and businesses understand customer feedback.

## 📁 Project Structure
**SentimentAnalysis-AmazonReviews/**
- **main.py** - Main controller that runs all experiments
- **project1.py** - Core algorithms (Perceptron, Average Perceptron, Pegasos)
- **utils.py** - Helper functions for data loading and plotting
- **test.py** - Testing script for debugging
- **stopwords.txt** - Common words to ignore during analysis
- **reviews_train.tsv** - Training data (3,500 reviews)
- **reviews_val.tsv** - Validation data (500 reviews)
- **reviews_test.tsv** - Test data (1,000 reviews)

**How It Works:**
1. Read reviews and break them into individual words
2. Count word occurrences to create "word fingerprints"
3. Train machine learning algorithms to recognize patterns
4. Predict sentiment of new, unseen reviews

## 🚧 Challenges Faced
| Challenge | Solution |
|-----------|----------|
| Converting text to numbers | Used "Bag of Words" technique - each word becomes a numerical feature |
| Too many unique words | Filtered 126 common stopwords (the, and, is, etc.) |
| Preventing overfitting | Used regularization in Pegasos algorithm |
| Choosing the best algorithm | Tested 3 approaches, Pegasos performed best |
| Binary vs count features | Binary (yes/no) features beat count (frequency) features |

## 📊 Performance Summary
### **Accuracy Results:**
| Model | Validation Accuracy | Test Accuracy | Key Feature |
|-------|---------------------|---------------|-------------|
| **Pegasos** | 80.6% | 80.8% | ⭐ **BEST** with stopwords removed |
| Average Perceptron | 80.0% | - | Good baseline performance |
| Perceptron | 79.4% | - | Simple but effective |

### **Feature Engineering Impact:**
- **Baseline (binary features):** 80.2%
- **With stopwords removed:** 80.8% (+0.6% improvement)
- **With word counts:** 77.0% (-3.2% decrease)

## 💡 Key Insights
1. **Simple beats complex**: Basic word presence (binary) worked better than counting word frequencies
2. **Punctuation matters**: The exclamation mark "!" was the 3rd strongest positive indicator
3. **Stopwords removal helps**: Filtering common words like "the", "and" improved accuracy
4. **Food-specific signals**: "Delicious" was the #1 most positive word
5. **Emphasis indicators**: ALL-CAPS words often signal strong sentiment

## 🌍 Real World Applications

### **For Consumers:**
- **Review summarizer**: "1,243 reviews: 85% positive, 15% negative"
- **Smart filtering**: "Show me negative reviews mentioning 'spicy'"
- **Trend alerts**: "Positive reviews for this cereal increased 40% this month"

### **For Businesses:**
- **Quality monitoring**: Auto-detect product issues from review patterns
- **Competitor analysis**: Compare sentiment across similar products
- **Marketing insights**: "Customers love our chocolate flavor but hate the packaging"

### **For Researchers:**
- **Public opinion mining**: Track sentiment about food trends (keto, vegan, gluten-free)
- **Language analysis**: Study how emotion is expressed in online reviews
- **A/B testing**: Compare review sentiment after product changes

## 🎓 Conclusion
This project demonstrates that **even simple machine learning** can solve real business problems. With just basic word counting and linear classifiers, we achieved **80%+ accuracy** in predicting review sentiment. The real power comes from understanding **what works** (binary features, stopword removal) and **what doesn't** (word counts).

Future improvements could include:
- **Emoji analysis** 😍 vs 😡
- **Sarcasm detection** ("Oh GREAT, another broken package")
- **Aspect-based sentiment** (separate ratings for taste, packaging, delivery)

## 👨‍💻 Author
**Rodrick** - Data Scientist/ML Engineer

As a passionate data scientist with a passion for turning raw data into actionable insights, I built this project to master the fundamentals of machine learning and natural language processing. What fascinates me most is how simple mathematical models can capture the complexities of human language and emotion. Beyond the code, I'm interested in how these techniques can create real business value - helping companies understand their customers better and helping consumers make more informed decisions. This project represents both technical learning and practical problem-solving, blending algorithm implementation with real-world application.

