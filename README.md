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


To run the API using flask run command:

• Ensure that all required packages are installed. <br/>
• Download the code and the trained model file 'model.pkl' to a directory on your computer. <br/>
• Open the Anaconda Prompt and navigate to the directory containing the code and the model file. <br/>
• Set the FLASK_APP environment variable by running the command set FLASK_APP=my_flask_app.py on Windows or export FLASK_APP=my_flask_app.py on Linux/MacOS. <br/>
• Optionally set the FLASK_ENV environment variable to 'development' to enable debug mode by running the command set FLASK_ENV=development on Windows or export FLASK_ENV=development on Linux/MacOS. <br/>
• Run the command flask run to start the Flask server. <br/>
• The API will be accessible at http://localhost:5000/predict. <br/>
Usage:

• Send a POST request to http://localhost:5000/predict with an image file attached. <br/>
• The response will be a JSON object containing the predicted class of skin lesion. <br/>
Note:

• The API has been configured to accept requests only from http://localhost:3000. If you want to use a different origin, you need to modify the CORS configuration in the code. <br/>
• The model was trained to classify skin lesions into one of nine classes. The classes are listed in the code and the predicted class will be returned as a string. <br/>
