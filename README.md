# Content-Based Movie Recommendation System

This project is a **Content-Based Recommendation System** built entirely from scratch in Node.js. It was developed as a Proof of Concept (PoC) to demonstrate a deep understanding of Information Retrieval (IR) algorithms without relying on external machine learning libraries like `scikit-learn`.

## 🎯 Key Features
- **Pure JavaScript Algorithmic Implementation:** Implements the Vector Space Model (VSM), TF-IDF weighting, $L_2$ Euclidean Norms, and Cosine Similarity (Scalar Product) using core JS array methods.
- **Content-Based Filtering:** Recommends movies by comparing a user's explicit feedback profile (+1, -1, 0) against movie features (genres, year, rating).
- **"Thin Server" Architecture:** Uses Express.js to provide real-time recommendations and an interactive interface.

## 🛠️ Tech Stack
- **Backend:** Node.js, Express.js
- **Frontend Views:** EJS (Embedded JavaScript templates), HTML/CSS
- **Methodology:** Information Retrieval Models, Linear Algebra

## 🧠 Theory to Implementation
The mathematical and theoretical foundation of this project is extensively documented. To see exactly how the concepts of TF-IDF, Vector Normalization, and User Profiling are mapped to the code, please read the [Theory Implementation Guide](theory-implementation.md).

## 🚀 How to Run Locally

1. **Install Dependencies:**
   ```bash
   npm install
   ```

2. **Start the Application:**
   ```bash
   npm start
   # or
   node server.js
   ```

3. **View the Application:**
   Open your browser and navigate to `http://localhost:3000` (or the port specified in your console).

## 📊 Experimental Evaluation
The `experiments.js` file contains experimental scripts to test system behavior against edge cases (such as the "Cold Start" problem or sparsity) and evaluates the impact of feature fusion (combining categorical metadata with numerical data).
