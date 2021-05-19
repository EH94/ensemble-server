import io
import time
import keras
import numpy as np
from os.path import join
from PIL import Image
from base64 import encodebytes
from keras import backend as K
from keras.models import Model, Input
from keras.layers.merge import average
from flask import Flask, request, jsonify, send_from_directory

# The below provided fucntions will be used from yolo_utils.py
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes

# The below functions from the yad2k library will be used
from yad2k2 import main
from yad2k.models.keras_yolo import yolo_head, yolo_eval

app = Flask(__name__)

memberCounter = 0
imageCounter = 1

@app.route("/")
def home():
    return jsonify("Members in ensemble: " + str(memberCounter))

@app.route("/GetImages")
def get_images():
    global imageCounter

    # Get paths for the next X images (X fake and X real) to send to edge device
    fake_images = [join("images/fake/", str(f) + ".jpg") for f in range(imageCounter, imageCounter + 10)]
    real_images = [join("images/real/", str(f) + ".jpg") for f in range(imageCounter, imageCounter + 10)]

    images = fake_images + real_images
    imageCounter += 10

    encoded_imges = []
    for image_path in images:
        try:
            encoded_imges.append(get_response_image(image_path))
        except:
            continue

    return jsonify({'images': encoded_imges})

@app.route("/GetAggregatedModel")
def get_aggregated_model():
    if(memberCounter > 1):
        return send_from_directory("model_data/models/", "ensemble.h5", as_attachment=True)
    else:
        return jsonify("No ensemble model found")

@app.route("/AggregateModel", methods=["POST"])
def aggregate_model():
    if 'weights' not in request.files:
        return jsonify("No weights file provided")
    
    file = request.files['weights']
    file.save("./model_data/weights/weights.weights")
    
    weights = open("./model_data/weights/weights.weights", 'rb')
    
    # Get time when conversion starts
    conversionStartTime = time.clock()
    # Convert weights to model
    model = convert_weights_to_keras_model(weights)    
    # Get time when conversion has finnished
    conversionEndTime = time.clock()

    # Get time when initiated adding model to ensemble
    baggingStartTime = time.clock()
    # Add model to ensemble
    bagging_ensemble_model(model)
    # Ge time when model has been added to ensemble
    baggingEndTime = time.clock()

    totalTimeConversion = conversionEndTime-conversionStartTime
    totalTimeAggregation = baggingEndTime-baggingStartTime

    print("Conversion of weights to keras model ", memberCounter, ": ", " - Time to convert: ", totalTimeConversion)
    print("Aggregation of model ", memberCounter, ": ", " - Time to aggregate: ", totalTimeAggregation)

    return jsonify("Model has been added to the ensemble")

@app.route("/GetEnsemblePrediction", methods=["POST"])
def get_ensemble_prediction():
    if(memberCounter > 1):
        if 'image' not in request.files:
            return jsonify("No image provided")
        
        file = request.files['image']
        file.save("predictions/" + file.filename)

        ensemble = keras.models.load_model('model_data/models/ensemble.h5', compile=False)
        image_name, out_scores, result= make_prediction("predictions/", file.filename, ensemble)

        response = "Image: " + str(image_name) + "\nPrediction: " + str(result) + "\nConfidence: " + str(out_scores)
        return jsonify(response)
    else:
        return jsonify("No ensemble model found")

@app.route("/GetSinglePrediction", methods=["POST"])
def get_single_prediction():
    if(memberCounter != 0):
        if 'image' not in request.files:
            return jsonify("No image provided")
        
        file = request.files['image']
        file.save("predictions/" + file.filename)

        firstModel = keras.models.load_model('model_data/models/firstModel.h5', compile=False)
        image_name, out_scores, result= make_prediction("predictions/", file.filename, firstModel)

        response = "Image: " + str(image_name) + "\nPrediction: " + str(result) + "\nConfidence: " + str(out_scores)
        return jsonify(response)
    else:
        return jsonify("No model found")

def bagging_ensemble_model(model):
    global memberCounter

    if(memberCounter == 0):
        keras.models.save_model(model, "model_data/models/firstModel.h5")
        memberCounter += 1

        return

    inputs = Input(shape=(608, 608, 3))
    if(memberCounter == 1):
        firstModel = keras.models.load_model('model_data/models/firstModel.h5', compile=True)
        x3 = average([firstModel(inputs), model(inputs)])
    else:
        existingEnsembleModel = keras.models.load_model('model_data/models/ensemble.h5', compile=True)
        existingEnsembleModel.layers.pop()

        values = [layer(inputs) for layer in existingEnsembleModel.layers]
        values.append(model(inputs))

        x3 = average(values[1:])

    newEnsembleModel = Model(inputs=inputs, outputs=x3)
    newEnsembleModel.compile(optimizer="sgd", loss="binary_crossentropy", metrics=["accuracy"])
    newEnsembleModel.summary()

    keras.models.save_model(newEnsembleModel, "model_data/models/ensemble.h5")

    memberCounter += 1

def make_prediction(image_path, input_image_name, yolo_model):
    #Obtaining the dimensions of the input image
    input_image = Image.open(image_path + input_image_name)
    width, height = input_image.size
    width = np.array(width, dtype=float)
    height = np.array(height, dtype=float)

    #Assign the shape of the input image to image_shapr variable
    image_shape = (height, width)

    #Loading the classes and the anchor boxes that are provided in the model_data folder
    class_names = read_classes("model_data/yolo_classes.txt")
    anchors = read_anchors("model_data/yolo_anchors.txt")

    #Print the summery of the model
    yolo_model.summary()

    #Convert final layer features to bounding box parameters
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

    #Now yolo_eval function selects the best boxes using filtering and non-max suppression techniques.
    # If you want to dive in more to see how this works, refer keras_yolo.py file in yad2k/models
    boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)

    # Initiate a session
    sess = K.get_session()

    #Preprocess the input image before feeding into the convolutional network
    image, image_data = preprocess_image(image_path + input_image_name, model_image_size = (608, 608))

    #Run the session
    out_scores, out_boxes, out_classes = sess.run(
        [scores, boxes, classes],
        feed_dict={
            yolo_model.input: image_data,
            K.learning_phase(): 0
        }
    )

    #Print the results
    print('Found {} boxes for {}'.format(len(out_boxes), input_image_name))
    
    #Produce the colors for the bounding boxs
    colors = generate_colors(class_names)
    
    #Draw the bounding boxes
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    
    #Apply the predicted bounding boxes to the image and save it
    image.save("predictions/" + input_image_name, quality=90)
    
    if(len(out_classes) == 0):
        result = "No box found"
    elif (out_classes[0] == 0):
        result = "real"
    else:
        result = "fake"

    return input_image_name, out_scores, result

def convert_weights_to_keras_model(weights_file):
    weights_header = np.ndarray(
        shape=(4, ), dtype='int32', buffer=weights_file.read(20))
    print('Weights Header: ', weights_header)

    return main("./model_data/weights/yolov2.cfg", weights_file, str(memberCounter))

def get_response_image(image_path):
    image = Image.open(image_path, mode='r')
    byte_arr = io.BytesIO()
    image.save(byte_arr, format='PNG')
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii')

    return encoded_img