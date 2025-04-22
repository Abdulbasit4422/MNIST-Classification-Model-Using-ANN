# MNIST Digit Classification Using ANN 🤖🧠🇦🇮👾
_A Deep Learning Project Comparing Convoluted Neural Network (CNN) vs. Artificial Neural Network (ANN) Performance after being trained on the same dataset that undergone the same data cleaning and transformation_
<br><br><br>

## 📌 Problem Statement
The MNIST dataset is a foundational benchmark in machine learning, consisting of 70,000 handwritten digit images (0-9). While modern deep learning models can achieve high accuracy on this dataset, it remains valuable for understanding fundamental neural network architectures and their comparative performance.

Key Challenges: <br>
✔ Achieving high accuracy (>98%) with a simple ANN architecture<br>
✔ Comparing performance between **Artificial Neural Networks (ANN)** and **Convolutional Neural Networks (CNN)** <br>
✔ Understanding the trade-offs between model complexity and accuracy<br>
✔ Preventing overfitting while maintaining generalization capability <br>
<br><br><br>

## 🎯 Objectives
✔ Build and train an ANN model for MNIST digit classification<br>
✔ Compare performance between ANN and CNN architectures<br>
✔ Analyze computational efficiency differences between ANN and CNN<br>
✔ Achieve competitive accuracy (>98%) with a fully connected network<br>
✔ Visualize model predictions to understand decision boundaries.<br>
<br><br><br>

## 📊 Dataset Overview
- Source: Classical MNIST dataset.
- Size: 70,000 grayscale images (28x28 pixels).
- Split: <br>
   ✔ 60,000 training images<br>
   ✔ 10,000 test images<br>
- Classes: 10 (digits 0-9). <br>
#### 🔗 Dataset Reference: [Yann LeCun’s MNIST Page](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz)
<br><br><br>

## 🛠 Skills & Technologies Used
### 📌 Core Technologiess
- Python (Primary Language)
- Jupyter Notebook (Interactive Development)
- Git & GitHub (Version Control)


### 📌 Key Libraries
- Deep Learning:	PyTorch
- Data Handling:	NumPy, Pandas
- Visualization:	Matplotlib, Seaborn
- Model Evaluation:	Scikit-learn (Metrics)


### 📌 Model Architectures 
- Input Layer: Flattened 784-dimensional vector (28×28 pixels)
- Hidden Layers: <br>
  Dense layers with ReLU activation<br>
  Dropout layers for regularization<br>
- Output Layer: 10-way softmax classification
### 📌 Training Process
- Optimizer: Adam <br>
- Loss Function: Categorical Crossentropy <br>
- Metrics: Accuracy <br>
- Validation Split: 20% of training data <br>
- Early Stopping: Monitor validation loss <br>
<br><br><br>

## ⚠ Limitations & Challenges
1. Architectural Limitations: ANNs lack spatial awareness compared to CNNs
2. Parameter Efficiency: Fully connected layers require more parameters
3. Feature Extraction: Manual feature engineering may be needed for optimal performance
4. Computational Load: Memory requirements grow quickly with image size
<br><br><br>

## 🚀 Future Improvements
✅ Hyperparameter Optimization: Systematic search for optimal architecture<br>
✅ Advanced Regularization: Experiment with L1/L2 and dropout rates<br>
✅ Feature Engineering: PCA or other dimensionality reduction techniques<br>
✅ Hybrid Models: Combine ANN with simple convolutional layers<br>
✅ Quantization: Optimize model for edge deployment<br>
<br><br><br>

## 📊 Performance Comparison (ANN vs CNN)

Metric  |   	ANN	  |   CNN
--------------------------
Test Accuracy | 98.2%	|  99.1%
-------------------------------
Training Time |	Faster 
|  Slower
-------------------------------
Parameters	| More |	Fewer
-------------------------------
Spatial Awareness |	No  |	Yes
-----
<br><br><br>




## 📜 License
This project is open-source under the MIT License. Contributions & feedback welcome!
<br><br><br>

 
 ## 📬 Open to collaboration
 You can  create a pull request with detailed explanation if you would love to work more on this, or contact me through:
 - [Github](https://www.github.com/Abdulbasit4422).
 - [LinkedIn](https://www.linkedin.com/in/oyetunjiabdulbasitoyebamiji)
 - [X](https://mobile.x.com/Abdulbasitoyeb1)
 - [Facebook](https://www.facebook.com/abdulbasit.oyetunji?mibextid=ZbWKwL)
 - Gmail --> abdulbasitoyetunji88@gmail.com




