# Importing essential libraries and modules

from flask import Flask, render_template, request, Markup
import pandas as pd
import numpy as np
from utils.disease import disease_dic
from utils.disease import disease_cause_dic
from utils.fertilizer import fertilizer_dic
from utils.fertilizer import analysis_dic
from utils.information import information_dic
from utils.information import image_dic
import requests
import pickle
import config
import torch
import io
from PIL import Image
from torchvision import transforms
from utils.disease_class import disease_classes
from utils.model import ResNet9

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()


# Loading crop recommendation model

crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))


# ==================================================================================================
def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None


def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])

    """
    A PyTorch Tensor is basically the same as a numpy array: it does not know anything about deep learning or computational graphs or gradients, 
    and is just a generic n-dimensional array to be used for arbitrary numeric computation.
    """
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    """
    Returns a new tensor with a dimension of size one inserted at the specified position.
    If you look at the shape of the array before and after, you see that before it was (4,) and after it is (1, 4) (when second parameter is 0) and (4, 1) (when second parameter is 1). 
    So a 1 was inserted in the shape of the array at axis 0 or 1, depending on the value of the second parameter.
    That is opposite of np.squeeze() (nomenclature borrowed from MATLAB) which removes axes of size 1 (singletons).
    """

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction



# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

# render home page


@ app.route('/')
def home():
    title = 'Farmous'
    return render_template('index.html', title=title)

# render crop recommendation form page


@ app.route('/crop-recommend')
def crop_recommend():
    title = 'Farmous'
    return render_template('crop.html', title=title)

# render fertilizer recommendation form page


@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Farmous'
    return render_template('fertilizer.html', title=title)

# render crop information form page


@ app.route('/about_us')
def about_us():
    title = 'Farmous'
    return render_template('about-us.html', title=title)


@ app.route('/crop-information')
def crop_information():
    title = 'Farmous'
    return render_template('crop-information.html', title=title)


@app.errorhandler(404)
def not_found(e):
    title = 'Farmous'
    return render_template("404.html", title=title)
# render disease prediction input page


# ===============================================================================================
# RENDER PREDICTION PAGES

# render crop recommendation result page


@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Farmous'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # state = request.form.get("stt")
        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]
            image_url = image_dic[final_prediction]

            return render_template('crop-result.html', prediction=final_prediction, title=title, image_url=image_url)

        else:

            return render_template('try_again.html', title=title)

# render fertilizer recommendation result page


@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Farmous'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    df = pd.read_csv('Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    analysis = analysis_dic[key]
    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title, analysis=analysis)

# render disease prediction result page


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Farmous'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)

            disease = Markup(str(disease_dic[prediction]))
            cause = Markup(str(disease_cause_dic[prediction]))
            return render_template('disease-result.html', prediction=disease, cause= cause, title=title)
        except:
            pass
    return render_template('disease.html', title=title)


@ app.route('/information', methods=['POST'])
def info_fetch():
    crop_name = str(request.form['cropname'])
    title = "Farmous"
    api_key = config.info_api_key
    base_url = "http://harvesthelper.herokuapp.com/api/v1/plants/"

    crop_id = information_dic[crop_name]
    image_url = image_dic[crop_name]
    complete_url = base_url + crop_id + "?api_key=" + api_key
    response = requests.get(complete_url)
    crop_infor = response.json()
    return render_template('information-result.html', info=crop_infor, title=title, image_url=image_url)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
