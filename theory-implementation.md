# Theoretical vs. Practical Implementation Analysis

**Project:** Content-Based Recommendation System

## 1. Introduction
This document provides a comparative analysis between the theoretical concepts of Information Retrieval models and Recommendation Systems, and their practical application in this Node.js project. The objective is to demonstrate that the implementation is firmly grounded in academic materials and algorithmic purity.

## 2. Concept Mapping: Theory to Implementation

### 2.1. Vector Space Model (VSM) & Bag of Words
*   **Theoretical Concept:** The Vector Space Model (VSM) represents documents and queries as vectors in a high-dimensional space. The "Bag of Words" approach ignores grammar and word order, focusing only on term presence or frequency.
*   **Practical Implementation:** 
    *   Implemented in `recommender.js` via the `extractVocabulary()` and `vectorizeObjects()` functions. 
    *   The "vocabulary" consists of all unique movie genres across the dataset. Each movie is vectorized into an $M$-dimensional array where categorical features are binary (1.0 for presence, 0.0 for absence), mirroring the Bag of Words model adapted for categorical metadata.

### 2.2. TF-IDF (Term Frequency - Inverse Document Frequency)
*   **Theoretical Concept:** TF-IDF is a weighting scheme designed to reflect the importance of a term to a document within a corpus. The Inverse Document Frequency (IDF) penalizes common terms to highlight discriminative features.
*   **Practical Implementation:** 
    *   Implemented in `calculateTfidfAndNorms()`.
    *   Because movie genres rarely repeat within a single movie description, the Term Frequency (TF) is treated as a binary or normalized feature.
    *   The IDF is explicitly calculated using the mathematical formula: `Math.log10(n / df)`. This ensures that common genres like "Drama" have a lower impact on the recommendation score than niche genres like "Film-Noir".

### 2.3. Vector Normalization (Euclidean Distance / $L_2$ Norm)
*   **Theoretical Concept:** Vectors of different lengths can skew similarity metrics. Normalization ensures all vectors have a unit length of 1, allowing for robust inner-product calculations.
*   **Practical Implementation:** 
    *   Implemented in `calculateTfidfAndNorms()`.
    *   The Euclidean norm ($||X|| = \sqrt{\sum x_i^2}$) is calculated manually using `.reduce()` followed by `Math.sqrt()`. This fulfills the strict requirement to avoid external ML libraries.

### 2.4. Content-Based User Profiling
*   **Theoretical Concept:** Content-Based Filtering builds a profile for a user based on the features of items the user has rated explicitly.
*   **Practical Implementation:**
    *   Implemented in `calculateUserProfile()` and the rating routes in `server.js`.
    *   The system collects explicit feedback (+1 for Like, -1 for Dislike, 0 for Neutral).
    *   The User Profile vector is aggregated by multiplying the normalized movie vectors by the user's explicit rating and summing them. This directly reflects the concept of shifting the user vector towards liked concepts and away from disliked ones.

### 2.5. Similarity Scoring (Inner Product)
*   **Theoretical Concept:** To rank documents against a query (or a user profile), the inner product (scalar product) or Cosine Similarity is computed.
*   **Practical Implementation:**
    *   Implemented in the `recommend()` function.
    *   The final relevance score is calculated as the scalar product of three components across all dimensions $k$: the movie vector, the IDF vector, and the User Profile vector (`score += vector[k] * idfVector[k] * userProfile[k]`). The movies are then sorted descendingly by this score.

## 3. Step-by-Step Mathematical Mapping (Algorithm Stepping)

To prove that the application performs matrix transformations correctly, this section traces an example (5 objects, 4 categories) and maps it directly to the JavaScript implementation.

### Phase 1: Raw Matrix, Norms, and TF-IDF
The raw data is first vectorized. Then, the vector norms, Document Frequencies (DF), and Inverse Document Frequencies (IDF) are calculated.

| | Cat 1 | Cat 2 | Cat 3 | Cat 4 | Vector Norm ($L_2$) |
|---|---|---|---|---|---|
| **Object 1** | 1 | 0 | 1 | 1 | 1.732 |
| **Object 2** | 1 | 1 | 0 | 0 | 1.414 |
| **Object 3** | 0 | 0 | 1 | 1 | 1.414 |
| **Object 4** | 1 | 0 | 0 | 1 | 1.414 |
| **Object 5** | 0 | 1 | 0 | 0 | 1 |
| **DF** | 3 | 2 | 2 | 3 | |
| **IDF** | 2.523 | 2.699 | 2.699 | 2.523 | |

**Code Implementation:**
The norms are calculated using the `.reduce` method to sum the squares, and `.map` for the square root. The IDF uses the exact decimal logarithm formula:
```javascript
// IDF Calculation
const idfVector = dfVector.map(df => {
    if (df === 0) return 0.0;
    return Math.log10(n / df);
});

// Norm Calculation
const norms = movieVectors.map(vector => {
    const sumSq = vector.reduce((acc, val) => acc + val * val, 0);
    return Math.sqrt(sumSq); 
});
```

### Phase 2: Normalized Matrix and User Profile Aggregation
The matrix elements are divided by their respective row norms. The user evaluates the objects (+1, -1, 0). The User Profile is the sum of the scalar products between the normalized category vectors and the user's ratings.

| | Cat 1 | Cat 2 | Cat 3 | Cat 4 | Rating |
|---|---|---|---|---|---|
| **Object 1** | 0.577 | 0 | 0.577 | 0.577 | **1** |
| **Object 2** | 0.707 | 0.707 | 0 | 0 | **0** |
| **Object 3** | 0 | 0 | 0.707 | 0.707 | **1** |
| **Object 4** | 0.707 | 0 | 0 | 0.707 | **0** |
| **Object 5** | 0 | 1 | 0 | 0 | **-1** |
| **User Profile**| 0.577 | -1 | 1.284 | 1.284 | |

**Code Implementation:**
The code iterates precisely over the user ratings, normalizes the vector on-the-fly (`vector[k] / norm`), multiplies by the rating, and aggregates the result:
```javascript
// Transformation corresponding to User Profile calculation
for (let k = 0; k < numFeatures; k++) {
    const normalizedVal = norm > 0 ? vector[k] / norm : 0;
    userProfile[k] += normalizedVal * rating; 
}
```

### Phase 3: Final Recommendation Scoring
The final score is determined by the scalar product of three vectors for each object: the object's vector, the IDF, and the User Profile.

| | Score |
|---|---|
| **Object 1** | 4.713 |
| **Object 2** | -0.878 |
| **Object 3** | 4.743 |
| **Object 4** | 3.321 |
| **Object 5** | -2.699 |

**Code Implementation (`recommend`):**
The implementation maps directly to this mathematical summation across all dimensions $k$, proving the adherence to the required scalar product logic:
```javascript
// Final Calculation
movieVectors.forEach((vector, i) => {
    let score = 0.0;
    for (let k = 0; k < numFeatures; k++) {
        score += vector[k] * idfVector[k] * userProfile[k];
    }
    recommendations.push({ score, title: movies[i].title });
});
```

## 4. Justification of Implementation Choices
1.  **Algorithmic Purity:** The explicit choice to avoid external machine learning libraries was respected. By using pure JavaScript array methods (`.reduce`, `.map`), the system demonstrates a fundamental understanding of the underlying linear algebra.
2.  **Feature Fusion:** The system utilizes numerical properties scaled to $\langle0, 1\rangle$. The `vectorizeObjects()` function successfully appends normalized `year` and `rating` to the genre vectors. This prevents ranking ties and adds information density to the vectors.
3.  **Proof of Concept Architecture:** The "Thin Server" Node.js architecture with an in-memory `USER_RATINGS` object fits the Proof of Concept requirement. It provides instant real-time recalculations for demonstration purposes.

## 5. Terminology Mapping
Here is a clear table mapping the key technical terms used in this project to their standard theoretical counterparts:

| Term in Project | Theoretical Equivalent | Context / Meaning |
| :--- | :--- | :--- |
| **Content-Based Filtering** | Content-based filtering | Core paradigm; recommending items based on feature similarity to user preferences. |
| **Vector Model (VSM)** | Vector Space Model | Model representing documents as vectors in a dimensional space. |
| **Cold Start** | Cold start problem | Advantage of CB: new items can be recommended immediately without waiting for collaborative ratings. |
| **Bag of Words (BoW)** | Bag of Words | Method of text/category representation regardless of order. |
| **TF-IDF** | TF-IDF scheme | Term weighting scheme combining local frequency and global rarity. |
| **Inverse Frequency (IDF)** | Inverse Doc. Frequency | Logarithmic calculation of term rarity in the entire collection. |
| **Euclidean Norm ($L_2$)** | Euclidean Norm | Mathematical requirement for vector length normalization. |
| **User Profile** | User Profile | Vector description of user preferences derived from their history. |
| **Explicit Feedback** | Explicit feedback | Ratings that a user enters consciously (e.g., +1 / -1 ratings). |
| **Scalar Product** | Scalar product / Inner prod. | Operation to calculate the similarity (score) between two vectors. |
| **Top-K** | Top-k ranking | Algorithm for selecting the $k$ best results to display to the user. |
| **Overspecialization** | Overspecialization | Limitation of CB: the system only recommends items very similar to those already rated. |

## 6. Conclusion
The implementation is an accurate, functional reflection of mathematical and conceptual paradigms in Information Retrieval. The codebase serves as a mathematically pure, compliant "Proof of Concept" for Content-Based Filtering.
