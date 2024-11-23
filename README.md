# College Placement Prediction system

This repository contains an end-to-end data mining and machine learning workflow. It demonstrates the process of preprocessing data, conducting exploratory analysis, building machine learning models, and evaluating their performance.

## Features  

- **Data Preprocessing**  
  Handles missing values, duplicates, and prepares the dataset for analysis.  

- **Exploratory Data Analysis (EDA)**  
  Visual and statistical techniques to understand trends and patterns in the data.  

- **Machine Learning Models**  
  Implements and evaluates multiple predictive models.  

- **Visualization**  
  Generates insightful plots and graphs to enhance understanding and decision-making.  

## Getting Started  

### Prerequisites  

- **Python 3.7+**
- **Libraries**:  
  - `pandas`  
  - `numpy`  
  - `matplotlib`  
  - `seaborn`  
  - `scikit-learn`  

### Installation  

1. Clone the repository:  
   ```bash
   git clone https://github.com/kavyanjali-kodali/Placement-Prediction-system.git
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the Notebook using Jupyter:
   ```bash
   jupyter notebook Data_Mining.ipynb


## Results

### Model Performance

The project evaluated multiple machine learning models for their accuracy, precision, recall, F1-score, and other metrics. Below are the key highlights:

| Model                  | Accuracy | Precision | Recall  | F1-score | ROC     | Mean Cross-Validation Score |  
|------------------------|----------|-----------|---------|----------|---------|-----------------------------|  
| Logistic Regression (LR) | 78.3%    | 70.9%     | 84.6%   | 77.2%   | 79.1%  | 83.8%                       |  
| Support Vector Machine (SVM) | 73.3% | 65.6%     | 80.8%   | 72.4%   | 74.2%  | 81.4%                       |  
| K-Nearest Neighbors (KNN) | 65.0%    | 57.6%     | 73.1%   | 64.4%   | 65.9%  | 74.3%                       |  
| Decision Tree (DT)     | 80.0%    | 73.3%     | 84.6%   | 78.6%   | 80.5%  | 80.8%                       |  
| Random Forest (RF)     | 80.0%    | 71.9%     | 88.5%   | 79.3%   | 81.0%  | 87.9%                       |  
| Gradient Boosting (GB) | **81.7%** | 72.7%     | **92.3%**| **81.4%**| **82.9%**| 86.8%                       |  

### Key Observations

- **Gradient Boosting** achieved the highest performance with an accuracy of **81.7%** and recall of **92.3%**, making it the most effective model for this dataset.  
- **Decision Tree** and **Random Forest** models also showed strong results with accuracies of **80%**.  
- **K-Nearest Neighbors (KNN)** underperformed compared to other models, highlighting its limitations with this dataset.  

## Contributing

Contributions are welcome! Feel free to submit a pull request or raise an issue for suggestions or improvements.
