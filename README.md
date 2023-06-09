# E-commerce Product Categorization
## Project Description
The project aims to develop a machine learning model using Long Short-Term Memory (LSTM) for categorizing unseen products into four categories: "Electronics", "Household", "Books", and "Clothing & Accessories". The model is trained on a dataset obtained from Ecommerce Text Classification. The project involves data preprocessing, model development using TensorFlow, training, evaluation, and visualization using Tensorboard. The goal is to develop a machine learning model using LSTM that achieves an accuracy of more than 85% and an F1 score of more than 0.7.
## Project Functionality
- Utilizes LSTM model to classify product texts into categories.
- Handles challenges related to data preprocessing and text classification.
- Saves the trained model and tokenizer for future use.
- Visualizes the model graph and training progress using Tensorboard.
## Challenges and Solutions
### Challenge: 
Data Preprocessing and Cleaning
### Solution: 
Pandas library is used to read and process the data, performing necessary data cleaning and exploratory data analysis.
### Challenge: 
Text classification and achieving desired accuracy and F1 score
### Solution: 
An LSTM model is developed using TensorFlow, with appropriate architecture and hyperparameter tuning, to achieve an accuracy of more than 85% and an F1 score of more than 0.7.
## Future Implementations
- Implementing additional features for improved model performance.
- Enhancing the preprocessing pipeline for better text representation.
- Exploring advanced LSTM architectures and techniques for better classification accuracy.
## How to Install and Run the Project
### 1. Clone the repository to your local machine using the following command:
```shell
git clone https://github.com/farah2p/farah-capstone2-ecommerce-text.git
```
### 2. Before running the code, ensure that you have the following dependencies installed:
- TensorFlow
- Pandas 1.5.3
- Matplotlib
- Tensorboard 2.12.3

Install the required dependencies by running the following command:
```shell
pip install tensorflow==2.12.0
pip install numpy==1.24.2
pip install matplotlib==3.7.1
pip install pandas==1.5.3
pip install tensorboard===2.12.3
```
### 3. Download the dataset from the provided Kaggle link and place it in the project directory.
### 4. Open the Jupyter Notebook or Python script containing the code.
### 5. Run the code cells or execute the script to perform data preprocessing, model training, and evaluation.
### 6. Use Tensorboard to visualize the model graph and training progress by running the following command in the project directory:
```shell
tensorboard --logdir tensorboard_logs/capstone2
```
Access Tensorboard in your web browser using the provided URL.
### 7. The trained model will be saved in the "saved_models" folder in .h5 format as model.h5
## Project Requirements
- Python
- TensorFlow library
## Usage
- Download the dataset from https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification and place it in the data directory.
- Preprocess the data using the provided Jupyter notebook farah-capstone2-ecommerce.ipynb.
- Train the LSTM model using the notebook farah-capstone2-ecommerce.ipynb. The notebook includes steps for training, evaluating and saving the model.
- To visualize the training process and performance, you can launch TensorBoard using the following command:
  tensorboard --logdir capstone_assignment_2
- Access the TensorBoard dashboard by opening a web browser and navigating to the provided URL.
- To categorize unseen products using the trained model, you can use the notebook farah-capstone2-ecommerce.ipynb. It provides an example of how to load the saved model and perform predictions on new data.
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
