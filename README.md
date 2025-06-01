# prompt: write a professional README file for this code

# 3D Head Pose Estimation

This project implements a robust and efficient system for 3D head pose estimation (Pitch, Yaw, and Roll), utilizing machine learning techniques trained on the AFLW2000 dataset. The system uses facial landmarks extracted by MediaPipe as features to predict the head orientation.

## Features

*   *Facial Landmark Extraction:* Utilizes the powerful MediaPipe Face Mesh model to accurately extract 468 3D facial landmarks.
*   *Pose Angle Loading:* Reads ground truth Pitch, Yaw, and Roll angles from the AFLW2000 dataset's .mat annotation files.
*   *Data Preprocessing:* Loads and prepares the landmark and pose data for model training.
*   *Statistical Analysis:* Provides visualizations and statistics of the pose angle distributions in the dataset.
*   *Multiple Regression Models:* Trains and evaluates various machine learning models for predicting pose angles, including:
    *   Linear Regression
    *   Ridge Regression
    *   Lasso Regression
    *   Elastic Net Regression
    *   Decision Tree Regression
    *   Support Vector Regression (SVR)
    *   Random Forest Regression
    *   XGBoost
*   *Model Evaluation:* Evaluates models using R², Mean Squared Error (MSE), and Mean Absolute Error (MAE).
*   *Best Model Selection:* Identifies and utilizes the best-performing model for each pose angle (Pitch, Yaw, Roll).
*   *Pose Axis Visualization:* Draws 3D coordinate axes on the face image to visually represent the predicted head pose.
*   *Testing:* Includes functions to test the model on single images and video files.

## Prerequisites

*   Python 3.7+
*   Google Colaboratory environment (recommended) or a Jupyter Notebook environment with necessary libraries installed.
*   Access to Google Drive for storing the dataset.

## Setup

1.  *Open in Google Colab:* The code is designed to run seamlessly in Google Colaboratory. Upload the .ipynb file to your Google Drive and open it with Google Colab.
2.  *Mount Google Drive:* Run the cell containing drive.mount('/content/drive') to connect your Google Drive. Grant the necessary permissions.
3.  *Download AFLW2000 Dataset:* Download the AFLW2000 dataset and upload the AFLW2000 folder containing the .jpg images and .mat annotation files to your Google Drive. **Ensure the path in the code (DATASET_PATH) correctly points to this folder.**
4.  *Install Dependencies:* Run the cell containing !pip install mediapipe.
5.  *Upload Test Images/Videos:* If you want to test on files other than those in the dataset, upload your test images (.jpg) and videos (.mov, .mp4, etc.) directly to the Colab environment's file system or a location accessible by the notebook.

## Usage

1.  *Run Cells Sequentially:* Execute each code cell in the notebook from top to bottom.
2.  *Data Loading and Preprocessing:* The notebook will automatically load the data, extract landmarks, and prepare the features and targets. This might take some time depending on the size of the dataset and the Colab runtime's performance.
3.  *Visualization and Analysis:* The notebook will display sample images with detected landmarks and pose axes, along with statistical plots for the pose angle distributions.
4.  *Model Training:* The defined machine learning models will be trained and evaluated on the dataset. The results will be printed.
5.  *Testing:*
    *   Modify the paths in the test_on_single_image() function calls to point to the images you want to test.
    *   Modify the paths in the test_on_video() function calls to point to the video you want to test.
    *   Run the respective testing cells. The results will be displayed visually.

## Code Structure

*   *Import Libraries:* Necessary libraries are imported at the beginning.
*   *Configuration:* DATASET_PATH is defined.
*   *Helper Functions:*
    *   draw_axis(): Draws 3D axes on an image.
    *   extract_landmarks(): Uses MediaPipe to get face landmarks.
    *   load_pose_angles(): Reads pose data from .mat files.
    *   draw_axis_function(): Modified draw_axis to originate from the detected nose point.
*   *Data Loading and Preprocessing:* Code to iterate through the dataset, extract features (landmarks), and load targets (pose angles).
*   *Landmark Visualization:* Code to display sample images with landmarks and ground truth pose axes.
*   *Statistical Analysis:* Code to perform and visualize statistical analysis of the pose data.
*   *Model Training and Evaluation:* Defines models, splits data, scales features, trains models, and evaluates their performance. Selects the best model for each angle.
*   *Prediction Function:* predict_pose_from_image() takes an image path and trained models to predict pose angles.
*   *Testing Functions:*
    *   test_on_single_image(): Tests prediction and visualization on a single image.
    *   test_on_video(): Tests prediction and visualization on a video file.
