from flask import Flask, Response, render_template, jsonify, session, redirect, url_for, request

# pip install Flask-WTF WTForms
from cProfile import label
from decimal import ROUND_HALF_UP, ROUND_UP
from wsgiref.validate import validator
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField, IntegerRangeField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename

import os

import cv2 
from hasconfCustom import video_detection, video_detection2
from flask_bootstrap import Bootstrap

app = Flask(__name__)

Bootstrap(app)
app.config['SECRET_KEY'] = 'mosharaf12345'
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


class UploadFileForm(FlaskForm):
    file = FileField('Upload File', validators=[DataRequired()])
    conf_slide = IntegerRangeField('Confidence Threshold', default=25, validators=[DataRequired()])
    submit = SubmitField('Submit')

# Global variables to store metrics
current_fps = 0
current_size = "0x0"
current_detections = 0  # This should match 'dpf' from your generator
total_detections = 0    # This should match 'total_detections' from your generator

def generate_frames(path_x = '', conf_ = 0.25):
    global current_fps, current_size, current_detections, total_detections
    
    rtsp_url = "rtsp://username:password@IP:PORT/cam/realmonitor?channel=5&subtype=0"
    video = cv2.VideoCapture(rtsp_url)
    yolo_output = video_detection2(path_x=rtsp_url, conf_=conf_) ## For RTSP stream

    # yolo_output = video_detection(path_x=path_x, conf_=conf_)  ## For local video file
    for detection_, fps_value, size_info, total_detections_count in yolo_output:
        ret, buffer = cv2.imencode('.jpg', detection_)
        current_fps = fps_value
        current_size = f"{size_info[1]}x{size_info[0]}"  # width x height
        current_detections = total_detections_count  # This is the cumulative total
        total_detections = total_detections_count    # Same value for now
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        

@app.route('/')
def home():
    return render_template('root.html')


# ---- Main route ----
@app.route('/frontPage', methods=['GET', 'POST'])
def frontPage():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        conf_value = form.conf_slide.data / 100.0
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Save values to session
        session['video_path'] = os.path.abspath(file_path)
        session['conf_value'] = conf_value

        # Redirect or render detection page
        return render_template('video.html')

    # If GET or invalid POST, show form
    return render_template('frontPage.html', form=form)

@app.route('/video_feed')
def video_feed():
    video_path = session.get('video_path')
    if not video_path or not os.path.exists(video_path):
        return "No video uploaded", 404

    return Response(generate_frames(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
    
@app.route('/video')
def video():
    return Response(generate_frames(path_x='data/sample_videos/football1.mp4', conf_=0.5), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/fpsgenerate')
def fps_fun():
    return jsonify(result=current_fps)

@app.route('/sizegenerate')
def size_fun():
    return jsonify(imageSize=current_size)

@app.route('/detectCount')
def detect_fun():
    return jsonify(detectCount=current_detections)

@app.route('/resetCounters')
def reset_counters():
    global current_fps, current_size, current_detections, total_detections
    current_fps = 0
    current_size = "0x0"
    current_detections = 0
    total_detections = 0
    return jsonify(status="counters reset")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5005, debug=True)
    # app.run()

'''
Option 3 — Auto-free port dynamically

You can make Flask try to run on the next available port:
if __name__ == '__main__':
    import socket
    port = 5000
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                break
            port += 1
    app.run(debug=True, port=port)
''' 




