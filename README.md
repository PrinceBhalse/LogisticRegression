### 🔹 Step-by-step Summary:

1. **Import Libraries**: Loads Pandas and relevant modules from `sklearn`.
2. **Load Data**: Reads Titanic data from `titanic.csv`.
3. **Preprocess Data**:

   * Drops irrelevant columns like *PassengerId*, *Name*, etc.
   * Fills missing values with `0`.
4. **Split Features and Target**:

   * `X`: All features except `'Survived'`.
   * `y`: The target label `'Survived'`.
5. **Train-Test Split**: Splits the dataset into training and testing sets (80%-20%).
6. **Model Training**: Trains a **Logistic Regression** model.
7. **Prediction**: Predicts survival on the test set.
8. **Accuracy Evaluation**: Prints the model's accuracy on the test data.

