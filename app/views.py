import mimetypes
import os
import uuid
from base64 import b64encode

import keras
import numpy as np
import requests
import wikipedia
from flask import render_template, request, redirect, url_for, send_from_directory
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from werkzeug.utils import secure_filename
from wtforms import StringField, validators
from yaml import load, SafeLoader
import keras.utils as image

from app.__init__ import app
from hub.examples.image_retraining.reverse_image_search import reverseImageSearch

# no secret key set yet
SECRET_KEY = os.urandom(32)
app.config["SECRET_KEY"] = SECRET_KEY
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# All paths below are derived from the project's own location, so the app
# works no matter where it's checked out on disk (no more hardcoded
# developer-machine paths like "/Users/.../UoL CNN").
UPLOAD_DIR = os.path.join(APP_ROOT, "uploads")
MODEL_DIR = os.path.join(app.root_path, "..", "hub", "examples", "image_retraining")
MODEL_PATH = os.path.join(MODEL_DIR, "CNN_Model.h5")

app.config["UPLOADED_PHOTOS_DEST"] = UPLOAD_DIR
os.makedirs(UPLOAD_DIR, exist_ok=True)

photos = UploadSet("photos", IMAGES)
configure_uploads(app, photos)

CLASS_NAMES = [
    "asteroids", "earth", "elliptical", "jupiter", "mars", "mercury",
    "moon", "neptune", "saturn", "spiral", "uranus", "venus",
]

# populated once an image has been saved, used by the reverse-image-search route
imageBytes = None


class SelectImageForm(FlaskForm):
    image_url = StringField(
        "image_url",
        validators=[validators.Optional(), validators.URL()],
        render_kw={"placeholder": "Enter a URL"},
    )
    image_file = FileField(
        "file",
        validators=[
            validators.Optional(),
            FileAllowed(["jpg", "jpeg", "png"], "Invalid File"),
        ],
        render_kw={"class": "custom-file-input"},
    )


def clear_upload_dir():
    """Remove any previously uploaded images so /result always reads the latest one."""
    for file_name in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)


def save_image_from_url(url):
    """Download an image from a URL into the uploads folder and return its filename.

    Raises ValueError with a user-friendly message if the URL can't be
    downloaded or doesn't point at an image.
    """
    try:
        response = requests.get(url, timeout=10, stream=True)
    except requests.RequestException as exc:
        raise ValueError(f"Couldn't reach that URL ({exc}).") from exc

    if not response.ok:
        raise ValueError(f"That URL returned an error (HTTP {response.status_code}).")

    content_type = response.headers.get("Content-Type", "")
    if not content_type.startswith("image/"):
        raise ValueError("That URL doesn't point to an image.")

    extension = mimetypes.guess_extension(content_type.split(";")[0].strip()) or ".jpg"
    if extension == ".jpe":
        extension = ".jpg"
    filename = secure_filename(f"{uuid.uuid4().hex}{extension}")
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return filename


@app.route("/uploads/<filename>")
def get_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/", methods=["GET", "POST"])
def index():
    clear_upload_dir()
    global imageBytes
    form = SelectImageForm()

    if form.validate_on_submit():
        url = (form.image_url.data or "").strip()
        uploaded_file = form.image_file.data
        has_file = uploaded_file is not None and getattr(uploaded_file, "filename", "")

        if not url and not has_file:
            # Mirrors the client-side check in case JS is disabled or bypassed.
            form.image_url.errors.append("Please provide an image URL or upload a file.")
            return render_template("index.html", form=form)

        try:
            if has_file:
                filename = photos.save(uploaded_file)
            else:
                filename = save_image_from_url(url)
        except ValueError as exc:
            target_field = form.image_file if has_file else form.image_url
            target_field.errors.append(str(exc))
            return render_template("index.html", form=form)

        file_path = os.path.join(UPLOAD_DIR, filename)
        imageBytes = (filename, open(file_path, "rb"))
        return redirect(url_for("result"))

    return render_template("index.html", form=form)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/result")
def result():
    try:
        files = os.listdir(UPLOAD_DIR)
        if not files:
            raise FileNotFoundError("No image has been uploaded yet. Please go back and choose one.")
        first_file = files[0]
        with open(os.path.join(UPLOAD_DIR, first_file), "rb") as f:
            image_data = f.read()
        base64_data = b64encode(image_data).decode("utf-8")
    except Exception as e:
        return render_template("error.html", detail=str(e))

    try:
        celestial_object, labels = predict_answer()
    except Exception as e:
        return render_template("error.html", detail=str(e))

    title, properties, description = wiki(celestial_object, MODEL_DIR)
    return render_template(
        "result.html",
        image=base64_data,
        labels=labels,
        title=title,
        description=description,
        properties=properties,
    )


@app.route("/redirectToGoogle")
def redirectToGoogle():
    if not imageBytes:
        return render_template("error.html", detail="No image available to search for. Please classify an image first.")
    searchUrl = reverseImageSearch(imageBytes)
    return redirect(searchUrl, 302)


@app.route("/predict")
def predict_answer():
    files = os.listdir(UPLOAD_DIR)
    if not files:
        raise FileNotFoundError("No image has been uploaded yet. Please go back and choose one.")

    first_file_path = os.path.join(UPLOAD_DIR, files[0])

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "The trained model file is missing. Place your trained CNN_Model.h5 file at "
            f"{MODEL_PATH} and try again."
        )

    img_width = img_height = 256
    img = image.load_img(first_file_path, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # normalize pixel values to [0, 1]

    model = keras.models.load_model(MODEL_PATH)
    predictions = model.predict(x)

    predicted_class_index = np.argmax(predictions)
    predicted_class_label = CLASS_NAMES[predicted_class_index]
    predicted_probability = predictions[0][predicted_class_index]
    print(f"The predicted class is {predicted_class_label} with probability {predicted_probability:.2f}")

    predictions_100 = [[p * 100 for p in row] for row in predictions]
    predictions_dict = dict(zip(CLASS_NAMES, predictions_100[0]))
    labels_and_scores = list(predictions_dict.items())

    return predicted_class_label, labels_and_scores


# return title, statistics and summary
def wiki(celestial_object, cwd):
    ans = celestial_object
    with open(os.path.join(cwd, "display_info.yml"), "r") as stream:
        all_display_statistics = load(stream, Loader=SafeLoader)

    req_statistics = all_display_statistics.get(ans, {})
    statistics = None
    title = None
    summary = None
    if ans in ["spiral", "elliptical"]:
        title = "Classified Celestial Object is {} Galaxy : ".format(ans.capitalize())
        summary = wikipedia.WikipediaPage(title="{} galaxy".format(ans)).summary
    elif ans in [
        "mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune",
    ]:
        title = "Classified Celestial Object is {} Planet : ".format(ans.capitalize())
        statistics = req_statistics.items()
        summary = wikipedia.WikipediaPage(title="{} (planet)".format(ans)).summary
    elif ans == "moon":
        statistics = req_statistics.items()
        summary = wikipedia.WikipediaPage(title="{}".format(ans)).summary
        title = "Classified Celestial Object is the {} : ".format(ans.capitalize())
    elif ans == "asteroids":
        statistics = req_statistics.items()
        summary = wikipedia.WikipediaPage(title="{}".format(ans)).summary
        title = "Classified Celestial Object is the {} : ".format(ans.capitalize())
    return title, statistics, summary
