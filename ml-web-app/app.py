import os
import EmailCleaner
import TermFrequencyInverseDocumentFrequency
import SupportVectorMachine
import numpy as np
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import sys
from pathlib import Path
from config import UPLOAD_FOLDER, REMOVE_TIME
from utils import allowed_file, get_current_time, get_model, preprocess, process, get_image_w_h, remove_old_files

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

try:
    sys.path.remove(str(parent))
except ValueError:
    pass


def run_model():
    corpus = []
    vals = []
    emails_to_predict = []

    loaded_svm_model = get_model()
    for email in os.listdir('static/uploads'):
        email_text = 'static/uploads/' + email
        with open(email_text, "r") as file:
            text = file.read()
        emails_to_predict.append(text)  # Used later to show the email content.
        return_string = EmailCleaner.email_reduction(email_text)  # Preprocess the email.
        corpus.append(return_string)  # Add the reduced email to a list of emails that will be labeled.
        val = TermFrequencyInverseDocumentFrequency.tf_idf([return_string]) # TF-IDF for a single email.
        vals.append(val[0])  # TF-IDF matrix is a list of lists, so take the first list.
    feature_vectors = feature_engineering(vals, 194)  # 194 because it was the longest length email in training set.

    predictions = loaded_svm_model.predict(feature_vectors)  # Get prediction.
    for i, prediction in enumerate(predictions):
        email_text = emails_to_predict[i]  # Load email content to be printed.
        if prediction == 0:
            formatted_text = f"Predicted Label: Scam"
        elif prediction == 1:
            formatted_text = f"Predicted Label: Non-scam"
        email_text = f"Email Text: {email_text}"

    return formatted_text, email_text


def feature_engineering(vals, max_email_length):
    feature_vectors = []

    for email in vals:
        email_length = len(email)

        # Pad the email to max_email_length with zeros (or any padding value)
        padding = np.zeros(max_email_length - email_length)
        padded_email = np.concatenate((email, padding))

        # Create binary padding indicators (1 for data, 0 for padding)
        padding_indicators = np.concatenate((np.ones(len(email)), np.zeros(len(padding))))

        # Combine the email tokens and padding indicators into a single list
        combined_features = np.concatenate((padded_email, padding_indicators))

        # Append the combined feature vector to the list
        feature_vectors.append(combined_features)

    return feature_vectors


@app.route('/')
def upload_form():
    return render_template('home.html')


@app.route('/retrain', methods=['POST'])
def retrain_model():
    error = None
    if request.method == 'POST':
        if request.form['input_field'] > '194':
            corpus = []
            labels = []
            vals = []

            print('In Progress...')
            # Preprocess scam emails then convert them to number matrices and add it to final list of values to be run on.
            for email in os.listdir('static/scam_emails'):
                email_text = "static/scam_emails/" + email
                return_string = EmailCleaner.email_reduction(email_text)  # Preprocess the email.
                corpus.append(return_string)  # Add the reduced email to a list of emails.
                val = TermFrequencyInverseDocumentFrequency.tf_idf([return_string])  # TF-IDF for a single email.
                vals.append(val[0])  # TF-IDF matrix is a list of lists, so take the first list.
                labels.append(1)  # 1 for scam.

            print('In Progress...')
            # Preprocess nonscam emails then convert them to number matrices and add it to final list of values to be run on.
            for email in os.listdir('static/nonscam_emails'):
                email_text = 'static/nonscam_emails/' + email
                return_string = EmailCleaner.email_reduction(email_text)  # Preprocess the email.
                corpus.append(return_string)  # Add the reduced email to a list of emails.
                val = TermFrequencyInverseDocumentFrequency.tf_idf([return_string])  # TF-IDF for a single email.
                vals.append(val[0])  # TF-IDF matrix is a list of lists, so take the first list.
                labels.append(0)  # 0 for nonscam.

            max_email_length = int(request.form['input_field'])  # Need to get the longest email length.
            vals = feature_engineering(vals, max_email_length)

            accuracy, report = SupportVectorMachine.svm(vals, labels)  # create model.
            flash('The model has been successfully retrained. Here are the results:')
            msg = f'Accuracy: {accuracy}'
            flash(msg)
            return render_template('model.html', tables=[report.to_html()])

        elif request.form['input_field'] < '194':
            error = 'Enter in a number greater than 194'
            return render_template('model.html', error=error)


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    curr_file = request.files['file']

    if curr_file.filename == '':
        flash('Invalid file')
        return redirect(request.url)

    if curr_file and allowed_file(curr_file.filename):
        remove_old_files(UPLOAD_FOLDER, REMOVE_TIME)

        now = get_current_time()
        filename = now + secure_filename(curr_file.filename)
        curr_file.save(os.path.join(UPLOAD_FOLDER, filename))

        prediction, email_text = run_model()

        flash(prediction)
        flash(email_text)

        filename = os.path.join(now + os.path.basename(filename))
        return render_template('home.html', filename=filename)

    else:
        flash('Allowed upload types are .txt files')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static',
                    filename='uploads/' + filename, code=301))


@app.route('/about', methods=['GET', 'POST'])
def about_page():
    if request.method == 'POST':
        return redirect(url_for('upload_form'))
    return render_template('about.html')


@app.route('/model', methods=['GET', 'POST'])
def model_page():
    if request.method == 'POST':
        return redirect(url_for('upload_form'))
    return render_template('model.html')


if __name__ == "__main__":
    app.run(port=5085)