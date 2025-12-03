# Character Identification via CNN

letters (A-Z) and numbers (0-9) in real-time. It uses a specific type of machine learning model called a Convolutional Neural Network (CNN), which is highly effective at image recognition.
The project logic is divided into three main components:
Data Preparation: It combines large datasets of handwritten letters and digits, resizing and formatting the images so the computer can process them.
Training: The code builds a neural network and "trains" it using the prepared data, teaching the system to distinguish between different characters based on visual patterns.
The Interface: A live application allows you to draw or crop characters on a digital whiteboard. The trained model creates a bounding box around your input and instantly predicts which character you wrote.

# Summarey of file

### **1. Data Preprocessing.ipynb**
This notebook handles the extraction, transformation, and loading (ETL) of the raw datasets.
* **Data Ingestion:** Loads handwritten letter data from a CSV file and digit data from the MNIST dataset.
* **Data Merging:** Concatenates the letter and digit datasets into a single repository containing approximately 442,000 images.
* **Formatting:** Reshapes all images to 28x28 pixels, normalizes pixel values (0-1 range), and converts labels into categorical formats.
* **Export:** Splits data into training/testing sets and saves them as Numpy files (`.npy`) for the model to use.

### **2. CNN Architecture.ipynb**
This notebook is responsible for constructing, training, and evaluating the neural network.
* **Model Construction:** Builds a Sequential Convolutional Neural Network (CNN) using layers such as `Conv2D` (for feature detection), `MaxPooling2D`, `BatchNormalization`, and `Dense` (for classification).
* **Training:** Trains the model over 10 epochs using the prepared Numpy data, optimizing for categorical cross-entropy loss.
* **Evaluation:** Validates the model against test data, achieving a final accuracy of approximately 99.01%, and generates a confusion matrix heatmap to visualize errors.
* **Saving:** Exports the best performing weights as a `.h5` file (`best_val_loss_model.h5`).

### **3. Application.ipynb**
This notebook serves as the user interface and deployment script.
* **Interface Creation:** Uses OpenCV (`cv2`) to generate a live window acting as a digital whiteboard.
* **Input Handling:** Tracks mouse events to allow users to draw characters directly on the screen.
* **Prediction Pipeline:** Loads the saved `.h5` model, captures the user's drawing, processes the image (resize/invert), and displays the top 3 character predictions with their confidence scores.

To run this project, you would typically follow a three-step process: **Data Preparation**, **Model Training**, and **Application Execution**.

---
# Project Requirments

Note: Go to "Test Folder" and run "Test File.ipynb" to check that every required package is available and access to dataset.

- Python 3.11
- Jupyter Notebook 7.5.0

## File Install
1) pip install notebook
2) python -m pip install --upgrade opencv-python numpy pandas matplotlib scikit-learn keras tensorflow
3) python -m pip install numpy visualkeras pandas seaborn keras matplotlib scikit-learn Pillow tensorflow
4) get-pip.py (Optional: Depend on your Enviroment or System)

----- Restart VS COde -----

### **Project Execution Steps** 

1.  **Prepare the Data (Run *Data Preprocessing.ipynb*):** This notebook must be executed first to create the necessary input files. It loads the letter and digit datasets, combines and standardizes the images to 28x28 pixels, and saves the final training and testing arrays as Numpy files (`.npy`).

2.  **Train the Model (Run *CNN Architecture.ipynb*):** Once the data files are ready, run this notebook to build the CNN model and teach it to recognize the characters. The script loads the `.npy` files, trains the network, and then saves the best-performing model weights to a file named `best_val_loss_model.h5`.

3.  **Launch the Application (Run *Application.ipynb*):** This is the final step. The *Application.ipynb* script imports the saved `best_val_loss_model.h5` file and initializes the graphical user interface using OpenCV (`cv2`). This interface allows you to start drawing characters, and the application will use the trained model to predict them in real-time.

You must run these notebooks in sequence, as the output of each file is required as input for the next.

