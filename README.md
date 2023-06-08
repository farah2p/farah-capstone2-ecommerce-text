# E-commerce Product Categorization
This project aims to categorize unseen products into four categories: Electronics, Household, Books, and Clothing & Accessories. The goal is to develop a machine learning model using LSTM that achieves an accuracy of more than 85% and an F1 score of more than 0.7.
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
## Saved Models
The trained model is saved in the saved_models directory in .h5 format. Additionally, the tokenizer used for text preprocessing is saved in .json format in the same directory.
## Results
You can find the screenshots of the model architecture, training process and performance reports in the results directory. These images provide an overview of the project's progress and results.
- Model performance:

![Model Performance](farah-model-performance.png)

- Model Architecture:

![Model Architecture](farah-model-summary.png)

- Tensorboard Accuracy:

![Tensorboard Accuracy](farah-accuracy-tensorboard.png)

- Tensorboard Loss:

![Tensorboard Loss](farah-loss-tensorboard.png)

- The trained model achieved a training accuracy of 97.65% and a validation accuracy of 95.18%. The low training loss of 0.0905 indicates that the model effectively minimized the difference between predicted and actual values during the training phase. Similarly, the validation loss of 0.2122 demonstrates good performance on unseen data.
- These results indicate that the model has learned to categorize products into the desired categories with high accuracy. However, it's important to note that there is a slight difference between the training and validation accuracies, suggesting a potential overfitting on the training data. Further fine-tuning and regularization techniques can be explored to improve generalization on unseen data.
- The F1 score obtained 0.952006628045495 indicates that the model has achieved a high level of accuracy in categorizing the products into the desired categories. This suggests that the model has a good balance between correctly identifying positive instances and avoiding false positives and false negatives.
- The model's performance meets the project requirements, achieving an accuracy of more than 85% and an F1 score of more than 0.7. The model's architecture and performance plots can be found in the directory and the trained model is saved in the 'saved_models' directory in .h5 format.
- For more details and instructions on how to run the project, please refer to the 'Usage' section in this README file.
## Credits
The dataset used in this project is sourced from Kaggle:
https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification

## Contributing
Contributions to this project are welcome. If you find any issues or have suggestions for improvement, please open an issue or submit a pull request on the GitHub repository.
