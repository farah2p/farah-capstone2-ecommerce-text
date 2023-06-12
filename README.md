# E-commerce Product Categorization

## Project Description
The project aims to develop a machine learning model using Long Short-Term Memory (LSTM) for categorizing unseen products into four categories: "Electronics", "Household", "Books", and "Clothing & Accessories". The model is trained on a dataset obtained from Ecommerce Text Classification. The project involves data preprocessing, model development using TensorFlow, training, evaluation, and visualization using Tensorboard. The goal is to develop a machine learning model using LSTM that achieves an accuracy of more than 85% and an F1 score of more than 0.7.

## Project Functionality
- Utilizes LSTM model to classify product texts into categories.
- Handles challenges related to data preprocessing and text classification.
- Saves the trained model and tokenizer for future use.
- Visualizes the model graph and training progress using Tensorboard.

## Challenges and Solutions
Some challenges faced during the project include:
- Data preprocessing: Handling missing values and duplicates, and performing text tokenization and padding.
- Model architecture selection: Choosing the appropriate architecture for text classification.
- Training and evaluation: Optimizing model performance and interpreting evaluation metrics.

To overcome these challenges, the project uses pandas for data handling, Keras for model creation, and scikit-learn for evaluation metrics. Extensive documentation and online resources were referenced to address specific challenges and ensure best practices.

## Future Challenges and Features
Some challenges and features that can be implemented in the future include:
- Handling imbalanced datasets: Implementing techniques to address class imbalance if encountered.
- Fine-tuning the model: Experimenting with hyperparameter tuning and exploring different architectures to improve performance.
- Multilingual support: Extending the model to handle multiple languages for broader applicability.

## File Structure
- ecommerceDataset.csv: The dataset containing text documents and their corresponding labels.
- main.py: The main Python script containing the code for the project.
- saved_models/: A directory to store the saved model in .h5 format.
- tokenizer.pkl: The tokenizer object saved using pickle for future use.
## Future Implementations
- Implementing additional features for improved model performance.
- Enhancing the preprocessing pipeline for better text representation.
- Exploring advanced LSTM architectures and techniques for better classification accuracy.

## Setup and Dependencies
To run the code, make sure you have the following dependencies installed:
- pandas
- numpy
- tensorflow
- matplotlib
- scikit-learn

## Getting Started
- Ensure that all the necessary dependencies are installed.
- Place the ecommerceDataset.csv file in the same directory as the main.py file.
- Run the main.py file to execute the code.
- The code will perform data preprocessing, train the model, evaluate its performance, visualize accuracy, and provide inference results.
- The trained model will be saved in the saved_models/ directory as model.h5, and the tokenizer object will be saved as tokenizer.pkl.

## How to Install and Run the Project
### 1. Clone the repository to your local machine using the following command:
```shell
git clone https://github.com/farah2p/farah-capstone2-ecommerce-text.git
```
### 2. Change to the project directory:
```shell
cd product-categorization
```
NOTES: Replace product-categorization with your project directory path
### 3. Before running the code, ensure that you have the installed all the required dependencies
Install the required dependencies by running the following command:
```shell
pip install tensorflow==2.12.0
pip install numpy==1.24.2
pip install matplotlib==3.7.1
pip install pandas==1.5.3
pip install tensorboard===2.12.3
```
### 4. Download the dataset:
- The dataset used for this project can be obtained from https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification 
- Place the downloaded dataset file (ecommerceDataset.csv) in the project directory.
### 5. Open the Jupyter Notebook or Python script containing the code.
Run the main script:
```shell
python farah-capstone2-ecommerce.ipynb
```

The script will perform data preprocessing, train the model, evaluate its performance, visualize accuracy, and provide inference results.
### 6. Use Tensorboard to visualize the model graph and training progress by running the following command in the project directory:
```shell
tensorboard --logdir tensorboard_logs/capstone2
```
Access Tensorboard in your web browser using the provided URL.
### 7. The trained model will be saved in the saved_models/ directory as model.h5, and the tokenizer object will be saved as tokenizer.pkl.

## Project Requirements
- Python
- TensorFlow library

## Results
After developing the LSTM-based model for E-commerce Product Categorization, the following results were achieved:
- Training Loss: 0.0905
- Training Accuracy: 97.65%
- Validation Loss: 0.2122
- Validation Accuracy: 95.18%
- F1 Score: 0.952

The project's objective was to develop a model using LSTM that achieves an accuracy of more than 85% and an F1 score of more than 0.7. The model successfully met and exceeded these requirements.

During training, the model achieved a low training loss of 0.0905, which considers as a good fit to the training data indicates that the model effectively minimized the difference between predicted and actual values during the training phase. The training accuracy reached an impressive 97.65%, demonstrating the model's ability to accurately classify E-commerce products into different categories. 

For validation, the model achieved a validation loss of 0.2122 and a validation accuracy of 95.18%. These results confirm that the model performs well not only on the training data but also on unseen validation data, showcasing its generalization capabilities.

Furthermore, the F1 score is an important metric that combines precision and recall. The model achieved an F1 score of 0.952, which suggests a high level of accuracy and effectiveness in categorizing the E-commerce products. It indicates that the model has achieved a high level of accuracy in categorizing the products into the desired categories. This suggests that the model has a good balance between correctly identifying positive instances and avoiding false positives and false negatives.

Overall, the developed LSTM-based model successfully met and surpassed the project's criteria, achieving an accuracy of over 85% and an F1 score of more than 0.7. This demonstrates the model's efficacy in accurately categorizing E-commerce products and provides a reliable solution for businesses seeking automated product classification.

Below are some sample visualizations generated by the project:

- Model performance:

![Model Performance](farah-model-performance.png)

- Model Architecture:

![Model Architecture](farah-model-summary.png)

- Tensorboard Accuracy:

![Tensorboard Accuracy](farah-accuracy-tensorboard.png)

- Tensorboard Loss:

![Tensorboard Loss](farah-loss-tensorboard.png)

## Credits
The dataset used in this project is sourced from Kaggle:
https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification

## Contributing
Contributions to this project are welcome. If you find any issues or have suggestions for improvement, please open an issue or submit a pull request on the GitHub repository.
