# Model
README for Flask API with PyTorch Model

This code is a Flask API that serves a PyTorch deep learning model trained to classify skin lesions. The API receives an image file via a POST request to the '/predict' endpoint, preprocesses the image, and passes it through the model to predict the class of skin lesion.

Prerequisites:

->Anaconda <br/>
->PyTorch <br/>
->Flask <br/>
->Flask-CORS <br/>
->PIL <br/>
->NumPy <br/>
->OpenCV <br/>


To run the API:

Ensure that all required packages are installed.
Download the code and the trained model file 'model.pkl' to a directory on your computer.
Open the Anaconda Prompt and navigate to the directory containing the code and the model file.
Run the command "python app.py" to start the Flask server.
The API will be accessible at http://localhost:8000/predict.
Usage:

Send a POST request to http://localhost:8000/predict with an image file attached.
The response will be a JSON object containing the predicted class of skin lesion.
Note:

The API has been configured to accept requests only from http://localhost:3000. If you want to use a different origin, you need to modify the CORS configuration in the code.
The model was trained to classify skin lesions into one of nine classes. The classes are listed in the code and the predicted class will be returned as a string.
