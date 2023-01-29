# Reducing-Wireless-Communication-Traffic--ANN
In this project, we implemented a Deep Learning (DL) Neural Network to manage network load effectiveness and network availability, employing in-network deep learning and prediction and created a "traffic type prediction" model. 

# Abstarct :
In this project, we implemented a Deep Learning (DL) Neural Network to manage network load effectiveness and network availability, employing in-network deep learning and prediction and created a "traffic type prediction" model. We train our model to monitor incoming traffic and estimate the network traffic type for an unidentified device type using accessible network Key Performance Indicators (KPIs). We can provide load balancing and make efficient use of the resources on the current network thanks to intelligent resource allocation. Even in the event of a network breakdown, our proposed model will be able to make wise decisions and choose the best network. This idea offers a brilliant method of reducing congestion on cellular and home Wi-Fi transmission stations by giving all essential and responsive services priority while routing all other services to the closest station

# Propsed Dataset:
Our dataset includes ( In Kaggel Platform(https://www.kaggle.com/datasets/puspakmeher/networkslicing)) the most important network and device KPIs, such as the type of connecting device (such as a smartphone, Internet of Things device, URLLC device, etc.), User Equipment (UE) category, Guaranteed Bit Rate, or GBR, packet delay budget, maximum packet loss, time and day of the week, etc. Control packets sent between the UE and network can be used to record these KPIs. All of this information is available because our model will operate internally on the network. Our system is being accessed by several distinct sorts of input devices. These include smartphones, IoT devices in general, AR-VR devices, Industry 4.0,  traffic e911 or public safety communication, healthcare, traffic for smart cities or smart homes, etc., or even an unidentified device requesting access to one or more services. These have UE category values assigned to them, and the network additionally gives each service request a pre-determined value. The model will also keep track of what hour and day of the week the system receives the request. Our ANN will keep track of all this data and utilize it to forecast future network resource reservations effectively and make wise decisions now.

# Creating a Multi-output classifier with Keras (Artificial neural network Multi-output classifier )


a-	Imports: Importing all of the necessary Python dependencies is the first step. Pandas and Numpy will be used along with the other four tools to read and process the dataset. Sklearn is generally used for activities relating to data pretreatment and preparation. We may import data from Sklearn and create train test split, which allows us to divide the data into a training dataset and a testing dataset. We will also use metrics for performance evaluations from Sklearn. Tensorflow for neural networks is the last. We will build our neural network using Dense (i.e. densely-connected) layers from the Sequential API of Tensorflow. We compute loss using categorical cross-entropy and optimize using SGD.To analyze the performance visually, the plot function from matplotlib will be used to plot the val accuracy and train accuracy curves. 

b-	Importing and Explore the Datasets: the next step is Importing and reading the datasets using pandas, CSV function, where the relevant dataset has been saved in CSV format  In addition, full details are presented about the type of datasets used to give a broad overview of its size in terms of the number of features and samples. To this end,   we print a quick overview of the data using data.info () function furthermore, using the data.head() code in order to display the first five rows and how our data frame looks. Using a function “unique”,to identify the number of categories\strings in the dataset can be known easily, in addition to repeating each category in all own columns.  Furthermore, by closely looking at the results, the data has four columns with categorical data and one in the output  The result of printing the” info” function is that the data set consists of four columns with a string categorical, and the same is the case with the outputs. Where it can be dealt with the categorical dataset using encoding techniques as will be explained later. 
.



c-	Identify Anomalies/ Missing Data: This dataset is KPIs data it can be captured from control packets between the UE and base station  Therefore, this dataset is real-time measured data, there are not any data points that immediately appear as anomalous, and no duplicating or missing data in any of the measurement columns. To verify our claim, the actual codes of the null and duplicate are written in the Report.Jupter file . 

d-	Features and Targets and Convert Data to Arrays The data must now be divided into features and targets. The class we wish to forecast is known as the target or label. As the target column represents traffic types (last column), which contains three different groups and classes that were previously mentioned, this project suggests eight features from the relevant dataset that will work in concert to forecast the last column, which represents the target output.
We will also convert the Pandas data frames to Numpy arrays because that is the way the algorithm works. For our dataset, select all rows and the first 8 columns (from 0 to 8) to X and all rows and the last column (from 8 to 10)  as y. As shown below. 

e-	Data Preparing: As mentioned earlier, the data set contains four different string values, three of which are at the input and one at the output. To solve this problem, in this project, several encoding methods will be proposed to deal with these values

-	Encode the Input Variable: The input variable contains four categorical data (4-column)  which refer to (Technology Supported,   	Day, 	GBR, and 	Use Case)  every one of them contains different string values. To manage this problem, the Label Encoder is adopted in this project to convert these string variables into labels\numeric. To this end, the LabelEncoder function is used as a code to solve this issue 
-	Encode the Output Variable:  When modeling multi-class classification problems using neural networks, it is good practice to reshape the output attribute from a vector that contains values for each class value to a matrix with a Boolean for each class value and whether a given instance has that class value or not. This is called one-hot encoding or creating dummy variables from a categorical variable. For example, in this problem, three class values are eMBB, URLLC, and mMTC. To clarify, before proceeding with encoding the output into binaries, it is necessary before this step to use the numerical encoder function (LabelEncoder) in order to convert it into numerical values in order to be dealt with by the accuracy matrix, as will be explained later.

f-	Data Scaling : The scaling step is one of the most important steps in creating neural networks, where the creation of weights for entries depends largely on scaling. To this end, and after selecting and preparing the features and targets, we scale them to a range between -1 and 1, using StandardScaler and fit it . By doing so, the overall datasets are scaled. 

g-	Train/test split :sklearn allows you to manually specify the dataset to use for validation\testing during training. After generating the dataset, we must create a split between training and testing data. Scikit-learn also provides a nice function for this: train_test_split.. Since both datasets are quite large, so in this project, we convert X and y into their training and testing components with an 80/20 train/test split. In other words, 80% of the data set samples will be used for training purposes, while 20% will be used for testing. It is worth noting that the random state here is one of the optimization elements that can be changed randomly depending on achieving a good performance of the model. According to the public domain, Random State is determined to 42 or 0. 

h-	Prepare the Neural Network Architecture: Now we will build a neural network that will contribute to training both datasets to predict single-label three different classes (eMBB, URLLC, and mMTC). Here the steps to build the network will be explained in detail:
a-	The next stage in this process is to build the model using a Sequential API instance. We then layer more tightly linked (Dense) layers on top using model.add. Remember from the above that each neuron in a layer links to every other neuron in the layer below it in a dense layer. This means that if any upstream neurons fire, they will become aware of certain patterns in those neurons.
b-	 The Input layer has the argument “input_dim” as an input dimensions, as the shape must equal the input data, then the value of input_dim is equal to 8.  
c-	The hidden layers : The dataset's eight inputs are sent through two hidden layers that were decided upon following some trial and error. As we approach the output layer, the neurons in our dense layers will get narrower. This enables us to identify numerous trends, which will improve the model's performance. If you're wondering how I arrived at the number of neurons in the hidden layers, I ran a number of tests and discovered that this number produces good results in terms of accuracy and error. As a result, the first and second are respectively built with 16 and 8 neurons. We employ ReLU as an activations function, as is typical.
d-	The output layer: Because we used a one-hot encoding for your network categories (eMBB, URLLC, and mMTC), the output layer must create three output values, one for each class. The output value with the largest value will be taken as the class predicted by the model. To this end,  a “softmax” activation function was used in the output layer. This ensures the output values are in the range of 0 and 1 and may be used as predicted probabilities. 


![image](https://user-images.githubusercontent.com/123154408/215338552-cd045b5f-223a-46d5-a76a-3b2085227888.png)




e-	Compiling the model: we then convert the model skeleton that we have just created into a true model. Using categorical_crossentropy as loss function (which can be used in effectively the number of multi-class tasks) and the SGD optimizer, we instantiate the model.
f-	Training the model: we then fit the training\learning  data to the model and provide a few configuration options defined earlier. The model will now start training with every epoch. Here, the number of epochs and the batch size are set in order to improve accuracy and performance, as these hyperparameters are very important in improving performance. It is worth noting that the validation of the performance of the model is determined through the validation data, which was previously set as part of the test data, where in each epoch the performance in terms of the matrices accuracy is evaluated and its quality is measured.

g-	Evaluating the model: after the model is trained, we can evaluate it using model. Predict. Based on the testing dataset, we then know how well it performs when it is used on data that it has never seen before.

h-	Model Performance Measurement: Tests of its final performance are now necessary. Additionally, the ANN are just another classification technique, thus you may use any classification statistic to evaluate the outcome. You could make use of the accuracy matrices score It is worth noting that this step takes place after decoding the three cases from being binary to numeric, i.e. converting it from One Hot Encoder form to form  Label Encoder. Then using accuracy matric to print the final result



# Tuning:
-	Use the optimizer SGD instead of Adam, Where it was noticed that the accuracy decreased and the effect of overfitting disappeared with the use of SGD due to its low learning rate, as shown in Figur below.  It is worth noting that the use of a number of optimization methods may negatively affect this in terms of increasing time and effort and making the model more complex.



![image](https://user-images.githubusercontent.com/123154408/215338614-09da66de-7432-41e7-9d38-aebc456d9852.png)  ![image](https://user-images.githubusercontent.com/123154408/215338628-00fc4f1c-2251-4600-bf2f-c0a04c73b918.png)
 
 
  The Effect of the optimizer types (1st Figure) Adam, overfitting .(2nd Figure) SGD , Normal fitting (proposed) 
 


