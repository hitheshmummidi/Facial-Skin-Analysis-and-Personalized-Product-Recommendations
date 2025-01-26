# # from flask import Flask, render_template, request
# # import tensorflow as tf
# # from tensorflow import keras
# # from keras.models import load_model
# # from keras.preprocessing import image
# # import numpy as np
# # import os


# # app = Flask(__name__)


# # # Load the trained model
# # model_path = 'models/my_model.h5'  # Adjust the path if necessary
# # if os.path.exists(model_path):
# #     model = load_model(model_path)
# # else:
# #     raise FileNotFoundError(f"Model file not found at {model_path}")


# # # Define a function to preprocess the image
# # def preprocess_image(img_path):
# #     img = image.load_img(img_path, target_size=(150, 150))
# #     img_array = image.img_to_array(img)
# #     img_array = np.expand_dims(img_array, axis=0)
# #     img_array /= 255.0
# #     return img_array

# # @app.route('/')
# # def home():
# #     return render_template('index.html')


# # @app.route('/predict',methods=['POST'])
# # def predict():
# #     if 'file' not in request.files:
# #         return redirect(request.url)
    
# #     file = request.files['file']
# #     if file.filename == '':
# #         return redirect(request.url)
    
# #     if file:
# #         file_path = os.path.join('uploads', file.filename)
# #         file.save(file_path)
        
# #         # Preprocess the image
# #         img_array = preprocess_image(file_path)
        
# #         # Make a prediction
# #         prediction = model.predict(img_array)
# #         predicted_class = np.argmax(prediction, axis=1)[0]
        
# #         # Map predicted class index to class name (adjust based on your class indices)
# #         class_indices = {0: 'Class 0', 1: 'Class 1', 2: 'Class 2'}  # Replace with your actual class names
# #         result = class_indices[predicted_class]
        
# #         return render_template('index.html', prediction=result)

# # if __name__ == "__main__":
# #     if not os.path.exists('uploads'):
# #         os.makedirs('uploads')
# #     app.run(host='0.0.0.0', port=8080, debug=True)

# from flask import Flask, render_template, request, redirect
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import os

# from ultralytics import YOLO
# import cv2

# app = Flask(__name__)

# # Load the trained model
# model_path = 'models/my_model.h5'  # Ensure this path is correct
# try:
#     model = load_model(model_path)
#     model.save('models/my_model_updated.h5')
# except Exception as e:
#     raise FileNotFoundError(f"Failed to load model at {model_path}. Error: {str(e)}")

# def preprocess_image(img_path):
#     img = image.load_img(img_path, target_size=(150, 150))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0
#     return img_array

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return redirect(request.url)
    
#     file = request.files['file']
#     if file.filename == '':
#         return redirect(request.url)
    
#     if file:
#         file_path = os.path.join('uploads', file.filename)
#         file.save(file_path)
        
#         img_array = preprocess_image(file_path)

#         # Load the YOLO model
#         model_yolo = YOLO('yolov8m.pt')  # Load the pre-trained YOLOv8 nano model
#         url_image = cv2.imread(file_path)
#         result_yolo = model_yolo(url_image)

#         for result in result_yolo:
#             for box in result.boxes:
#                 if int(box.cls[0]) == 0:
#                     print("human")
#                 else:
#                     print("object")
        
#         prediction = model.predict(img_array)
#         predicted_class = np.argmax(prediction, axis=1)[0]
        
#         class_indices = {0: 'Acne', 1: 'Clear', 2: 'Comedone'}
#         result = class_indices.get(predicted_class, 'Unknown condition')
        
#         return render_template('index.html', prediction=result)

# if __name__ == "__main__":
#     if not os.path.exists('uploads'):
#         os.makedirs('uploads')
#     app.run(host='0.0.0.0', port=8080, debug=True)
# from flask import Flask, render_template, request, jsonify
# import tensorflow as tf
# from tensorflow import keras
# from keras.models import load_model
# from keras.preprocessing import image
# import numpy as np
# import os
# import base64
# from PIL import Image
# import io

# app = Flask(__name__)

# # Load the trained model
# model_path = 'models/my_model.h5'  # Adjust the path if necessary
# if os.path.exists(model_path):
#     model = load_model(model_path)
# else:
#     raise FileNotFoundError(f"Model file not found at {model_path}")

# # Define a function to preprocess the image
# def preprocess_image(img_path):
#     img = image.load_img(img_path, target_size=(150, 150))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0
#     return img_array

# # Define a function to preprocess the image from base64 data
# def preprocess_image_from_base64(base64_str):
#     img = Image.open(io.BytesIO(base64.b64decode(base64_str)))
#     img = img.resize((150, 150))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0
#     return img_array

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' in request.files and request.files['file'].filename != '':
#         file = request.files['file']
#         file_path = os.path.join('uploads', file.filename)
#         file.save(file_path)
        
#         # Preprocess the image
#         img_array = preprocess_image(file_path)
#     elif 'image_data' in request.form:
#         base64_image = request.form['image_data']
#         img_array = preprocess_image_from_base64(base64_image)
#     else:
#         return jsonify({'error': 'No image provided'}), 400
    
#     # Make a prediction
#     prediction = model.predict(img_array)
#     predicted_class = np.argmax(prediction, axis=1)[0]
    
#     # Map predicted class index to class name (adjust based on your class indices)
#     class_indices = {0: 'Acne', 1: 'Clear Skin', 2: 'Comedone'}  # Replace with your actual class names
#     result = class_indices.get(predicted_class, 'Unknown Condition')
    
#     return jsonify({'prediction': result})

# if __name__ == "__main__":
#     if not os.path.exists('uploads'):
#         os.makedirs('uploads')
#     app.run(host='0.0.0.0', port=8080, debug=True)

# using this again
# from flask import Flask, render_template, request, jsonify
# import tensorflow as tf
# from tensorflow import keras
# from keras.models import load_model
# from keras.preprocessing import image
# import numpy as np
# import os
# import base64
# from PIL import Image
# import io

# app = Flask(__name__)

# # Load the trained model
# model_path = 'models/my_model.h5'  # Adjust the path if necessary
# if os.path.exists(model_path):
#     model = load_model(model_path)
# else:
#     raise FileNotFoundError(f"Model file not found at {model_path}")

# # Define a function to preprocess the image
# def preprocess_image(img_path):
#     img = image.load_img(img_path, target_size=(150, 150))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0
#     return img_array

# # Define a function to preprocess the image from base64 data
# def preprocess_image_from_base64(base64_str):
#     img = Image.open(io.BytesIO(base64.b64decode(base64_str)))
#     img = img.resize((150, 150))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0
#     return img_array

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' in request.files and request.files['file'].filename != '':
#         file = request.files['file']
#         file_path = os.path.join('uploads', file.filename)
#         file.save(file_path)
        
#         # Preprocess the image
#         img_array = preprocess_image(file_path)
#     elif 'image_data' in request.form:
#         base64_image = request.form['image_data']
#         img_array = preprocess_image_from_base64(base64_image)
#     else:
#         return jsonify({'error': 'No image provided'}), 400
    
#     # Make a prediction
#     prediction = model.predict(img_array)
#     predicted_class = np.argmax(prediction, axis=1)[0]
    
#     # Map predicted class index to class name (adjust based on your class indices)
#     class_indices = {0: 'Acne', 1: 'Clear Skin', 2: 'Comedone'}  # Replace with your actual class names
#     result = class_indices.get(predicted_class, 'Unknown Condition')
    
#     return jsonify({'prediction': result})

# if __name__ == "__main__":
#     if not os.path.exists('uploads'):
#         os.makedirs('uploads')
#     app.run(host='0.0.0.0', port=8080, debug=True)




# example 

# from flask import Flask, render_template, request, jsonify
# import tensorflow as tf
# from tensorflow import keras
# from keras.models import load_model
# from keras.preprocessing import image
# from sklearn.metrics.pairwise import cosine_similarity
# import pickle
# import numpy as np
# import os
# import base64
# from PIL import Image
# import io

# app = Flask(__name__)

# # Define the function to recommend products
# def recommend_products(normalized_user_profile, normalized_features, product_names, top_n=3):
#     # Compute cosine similarity between user profile and all products
#     user_sim = cosine_similarity(normalized_user_profile, normalized_features)

#     # Get the top N most similar products
#     top_indices = user_sim[0].argsort()[-(top_n+1):][::-1][1:]  # Skip the first one as it will be the user profile itself

#     # Get the product names of the top N recommendations
#     recommended_products = product_names.iloc[top_indices]

#     return recommended_products

# # Load the components
# def load_components():
#     with open('components(1).pkl', 'rb') as file:
#         return pickle.load(file)

# dataf, scaler, feature_columns, product_names = load_components()

# # Load the trained model
# model_path = 'models/my_model.h5'  # Adjust the path if necessary
# if os.path.exists(model_path):
#     model = load_model(model_path)
# else:
#     raise FileNotFoundError(f"Model file not found at {model_path}")

# # Define a function to preprocess the image
# def preprocess_image(img_path):
#     img = image.load_img(img_path, target_size=(150, 150))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0
#     return img_array

# # Define a function to preprocess the image from base64 data
# def preprocess_image_from_base64(base64_str):
#     img = Image.open(io.BytesIO(base64.b64decode(base64_str)))
#     img = img.resize((150, 150))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0
#     return img_array

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' in request.files and request.files['file'].filename != '':
#         file = request.files['file']
#         file_path = os.path.join('uploads', file.filename)
#         file.save(file_path)
        
#         # Preprocess the image
#         img_array = preprocess_image(file_path)
#     elif 'image_data' in request.form:
#         base64_image = request.form['image_data']
#         img_array = preprocess_image_from_base64(base64_image)
#     else:
#         return jsonify({'error': 'No image provided'}), 400
    
#     # Make a prediction
#     prediction = model.predict(img_array)
#     predicted_class = np.argmax(prediction, axis=1)[0]
    
#     # Map predicted class index to class name (adjust based on your class indices)
#     class_indices = {0: 'Acne', 1: 'Clear Skin', 2: 'Comedone'}  # Replace with your actual class names
#     result = class_indices.get(predicted_class, 'Unknown Condition')
    
#     # Get user profile information
#     skin_type = request.form.get('skin_type')
#     concerns = request.form.getlist('concerns')
    
#     # Prepare user profile for recommendation
#     user_profile = {
#         'Combination': 1 if skin_type == 'combination' else 0,
#         'Dry': 1 if skin_type == 'dry' else 0,
#         'Oily': 1 if skin_type == 'oily' else 0,
#         'Sensitive': 1 if skin_type == 'sensitive' else 0,
#         'Acne': 1 if 'acne' in concerns else 0,
#         'Irritation': 1 if 'irritation' in concerns else 0,
#         'Broken barrier': 1 if 'broken-barrier' in concerns else 0,
#         'Dark Spots': 1 if 'dark-spots' in concerns else 0,
#         'Exfoliation': 1 if 'exfoliation' in concerns else 0,
#         'Hydration': 1 if 'hydration' in concerns else 0,
#         'Pigmentation': 1 if 'pigmentation' in concerns else 0,
#         'Pimples': 1 if 'pimples' in concerns else 0,
#         'Pores': 1 if 'pores' in concerns else 0,
#         'Skin soothing': 1 if 'skin-soothing' in concerns else 0,
#         'Sun protection': 1 if 'sun-protection' in concerns else 0,
#         'Whitehead/Blackhead': 1 if 'whitehead-blackhead' in concerns else 0
#     }
    
#     return jsonify({'prediction': result, 'user_profile': user_profile})

# @app.route('/recommend', methods=['POST'])
# def recommend():
#     user_profile = request.json['user_profile']
#     normalized_user_profile = scaler.transform([list(user_profile.values())])
    
#     # Normalize the feature data
#     features = dataf[feature_columns]
#     normalized_features = scaler.transform(features)
    
#     recommended_products_list = recommend_products(normalized_user_profile, normalized_features, product_names)
    
#     return jsonify(recommended_products_list.tolist())

# if __name__ == "__main__":
#     if not os.path.exists('uploads'):
#         os.makedirs('uploads')
#     app.run(host='0.0.0.0', port=8080, debug=True)


# import os
# import numpy as np
# import pandas as pd
# from flask import Flask, request, jsonify, render_template
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.preprocessing import LabelEncoder
# import pickle

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads'

# # Load your models and encoders

# skin_condition_model = load_model('models/my_model_updated.h5')
# product_recommendation_model = pickle.load(open('components(1).pkl', 'rb'))
# label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

# product_data = pd.read_csv('Skinpro - Skinpro (2).csv.csv')

# def preprocess_image(img_path):
#     img = image.load_img(img_path, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0
#     return img_array

# def recommend_products(user_profile):
#     # Calculate cosine similarity between user profile and product data
#     product_profiles = product_data.drop(['Product', 'product_url', 'product_pic'], axis=1)
#     similarities = cosine_similarity([user_profile], product_profiles)[0]
#     top_indices = similarities.argsort()[-5:][::-1]
#     return product_data.iloc[top_indices]

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/analyze', methods=['POST'])
# def analyze():
#     name = request.form['name']
#     age = int(request.form['age'])
#     skin_type = request.form['skin-type']
#     concerns = request.form.getlist('concerns')
    
#     # Create user profile
#     user_profile = np.zeros(len(product_data.columns) - 3)
#     for concern in concerns:
#         index = list(product_data.columns).index(concern)
#         user_profile[index - 3] = 1
    
#     # Save uploaded file
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'})
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'})
    
#     if file:
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#         file.save(file_path)
        
#         # Preprocess the image
#         img_array = preprocess_image(file_path)
        
#         # Predict skin condition
#         predictions = skin_condition_model.predict(img_array)
#         predicted_class = label_encoder.inverse_transform([np.argmax(predictions)])
#         condition = predicted_class[0]
        
#         # Recommend products
#         recommendations = recommend_products(user_profile)
#         recommendations = recommendations.to_dict(orient='records')
        
#         return jsonify({'condition': condition, 'recommendations': recommendations})

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os
import base64
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
model_path = 'models/my_model.h5'  # Adjust the path if necessary
if os.path.exists(model_path):
    model = load_model(model_path)

else:
    raise FileNotFoundError(f"Model file not found at {model_path}")


# reading the data from dataset
dataf = pd.read_csv('preprocessed_dataset_products.csv')

# Define the unique feature columns
feature_columns = [
    'Combination', 'Dry', 'Oily', 'Sensitive', 'Acne', 'Irritation',
    'Broken barrier', 'Dark Spots', 'Exfoliation', 'Hydration',
    'Pigmentation', 'Pimples', 'Pores', 'Skin soothing', 'Sun protection',
    'Whitehead/Blackhead'
]
# Strip leading and trailing spaces from column names
dataf.columns = dataf.columns.str.strip()
# Ensure 'Product' column is string type
dataf['Product'] = dataf['Product'].astype(str)
# Define features and product names
features = dataf[feature_columns]
product_names = dataf['Product']

# Normalize the feature data
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)
# Get the product names
product_names = dataf['Product']

def get_user_profile(form_data):
    # Extract user input from form data
    user_input = {column: int(form_data.get(column, 0)) for column in feature_columns}

    # Create a DataFrame from the user input
    user_profile = pd.DataFrame([user_input], columns=features.columns)

    # Normalize the user profile using the same scaler
    normalized_user_profile = scaler.transform(user_profile)

    return normalized_user_profile


def recommend_products(normalized_user_profile, normalized_features, product_names, top_n=3):
    # Compute cosine similarity between user profile and all products
    user_sim = cosine_similarity(normalized_user_profile, normalized_features)

    # Get the top N most similar products
    top_indices = user_sim[0].argsort()[-(top_n+1):][::-1][1:]  # Skip the first one as it will be the user profile itself

    # Get the product names of the top N recommendations
    recommended_products = product_names.iloc[top_indices]

    return recommended_products


# Define a function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Define a function to preprocess the image from base64 data
def preprocess_image_from_base64(base64_str):
    img = Image.open(io.BytesIO(base64.b64decode(base64_str)))
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        
        # Preprocess the image
        img_array = preprocess_image(file_path)
    elif 'image_data' in request.form:
        base64_image = request.form['image_data']
        img_array = preprocess_image_from_base64(base64_image)
    else:
        return jsonify({'error': 'No image provided'}), 400
    
    # Make a prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    # Map predicted class index to class name (adjust based on your class indices)
    class_indices = {0: 'Acne', 1: 'Clear Skin', 2: 'Comedone'}  # Replace with your actual class names
    result = class_indices.get(predicted_class, 'Unknown Condition')
    
    return jsonify({'prediction': result})

@app.route('/recommend', methods=['POST'])
def recommend():
    form_data = request.form
    user_profile = get_user_profile(form_data)
    recommended_products = recommend_products(user_profile, normalized_features, product_names, top_n=3)
    
    return jsonify({'recommendations': recommended_products.tolist()})

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(host='0.0.0.0', port=8080, debug=True)
