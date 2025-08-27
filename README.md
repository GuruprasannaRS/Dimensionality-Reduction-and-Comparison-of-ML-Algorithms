# Dimensionality Reduction Impact on Machine Learning Classification Performance

This project demonstrates the application of dimensionality reduction techniques (PCA and LDA) and compares the performance of various machine learning algorithms on two datasets: Abalone and Wine Quality.

## Project Overview

The project explores how dimensionality reduction affects the performance of different classification algorithms by applying Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA) to reduce feature dimensions and then comparing classification accuracy across multiple algorithms.

## Datasets

### Abalone Dataset
- **Features**: Sex, Length, Diameter, Height, Whole weight, Shucked weight, Viscera weight, Shell weight
- **Target**: Rings (age classification)
- **Size**: No null values, categorical and numerical features

### Wine Quality Dataset
- **Features**: Fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol
- **Target**: Quality (1-10 scale)
- **Types**: Red wine and white wine combined
- **Size**: Combined dataset with quality labels

## Dimensionality Reduction Techniques

### Principal Component Analysis (PCA)
- **Abalone**: Applied to 9 features, optimal components: 5
- **Wine**: Applied to 12 features, optimal components: 8
- **Visualization**: Scree plots showing explained variance ratios
- **2D Visualization**: First two principal components plotted

### Linear Discriminant Analysis (LDA)
- **Abalone**: Applied to 9 features, optimal components: 3
- **Wine**: Applied to 12 features, optimal components: 4
- **Visualization**: Scree plots showing explained variance ratios
- **2D Visualization**: First two LDA components plotted

### t-SNE Visualization
- Applied to both PCA and LDA reduced datasets
- 2D visualization for pattern analysis

## Machine Learning Algorithms

### 1. K-Nearest Neighbors (KNN)
- **Abalone Raw**: 25.96% accuracy
- **Abalone PCA**: 26.32% accuracy (5 components)
- **Abalone LDA**: 27.27% accuracy (3 components)
- **Wine Raw**: 67.69% accuracy
- **Wine PCA**: 67.62% accuracy (8 components)
- **Wine LDA**: 66.08% accuracy (4 components)

### 2. Multinomial Naive Bayes
- **Abalone Raw**: 17.94% accuracy
- **Abalone PCA**: 15.19% accuracy
- **Abalone LDA**: 15.19% accuracy
- **Wine Raw**: 47.77% accuracy
- **Wine PCA**: 44.54% accuracy
- **Wine LDA**: 44.54% accuracy

### 3. Complement Naive Bayes
- **Abalone Raw**: 18.30% accuracy
- **Abalone PCA**: 17.58% accuracy
- **Abalone LDA**: 17.22% accuracy
- **Wine Raw**: 43.38% accuracy
- **Wine PCA**: 43.08% accuracy
- **Wine LDA**: 44.46% accuracy

### 4. Decision Trees
- **Abalone Raw**: 27.15% accuracy
- **Abalone PCA**: 24.16% accuracy
- **Abalone LDA**: 26.08% accuracy
- **Wine Raw**: 53.92% accuracy
- **Wine PCA**: 54.08% accuracy
- **Wine LDA**: 52.69% accuracy

### 5. Random Forest
- **Abalone Raw**: 26.56% accuracy
- **Abalone PCA**: 26.44% accuracy
- **Abalone LDA**: 27.87% accuracy
- **Wine Raw**: 61.46% accuracy
- **Wine PCA**: 60.08% accuracy
- **Wine LDA**: 57.62% accuracy

### 6. Gradient Boosting
- **Abalone Raw**: 26.20% accuracy
- **Abalone PCA**: 27.27% accuracy
- **Abalone LDA**: 27.63% accuracy
- **Wine Raw**: 60.62% accuracy
- **Wine PCA**: 57.46% accuracy
- **Wine LDA**: 56.46% accuracy

## Key Findings

1. **Dimensionality Impact**: PCA and LDA generally maintain or slightly improve classification accuracy while reducing computational complexity
2. **Algorithm Performance**: 
   - Wine dataset shows better overall performance than Abalone dataset
   - Random Forest and Gradient Boosting achieve the highest accuracies
   - Naive Bayes variants perform poorly on both datasets
3. **Optimal Components**: 
   - Abalone: 5 PCA components, 3 LDA components
   - Wine: 8 PCA components, 4 LDA components
4. **Execution Time**: Random Forest is significantly faster than Gradient Boosting

## Technical Implementation

- **Data Preprocessing**: MinMaxScaler for normalization, one-hot encoding for categorical variables
- **Cross-validation**: 5-fold cross-validation for hyperparameter tuning
- **Hyperparameter Optimization**: GridSearchCV for finding optimal parameters
- **Visualization**: Matplotlib and Seaborn for plots and heatmaps
- **Libraries**: scikit-learn, pandas, numpy, matplotlib, seaborn

## Project Structure

- `pca-lda-on-abalone-and-wine.ipynb`: Main Jupyter notebook containing all analysis
- `README.md`: Project documentation

## Requirements

The project uses standard data science libraries available in Python:
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
