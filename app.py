from flask import Flask, render_template , request , jsonify, Response, url_for
from PIL import Image
import os , io , sys
import numpy as np 
import cv2
import base64

from yolo_detection_images import runModel
from flask_bootstrap import Bootstrap

from object_detection import *
from camera_settings import *

app = Flask(__name__)
Bootstrap(app)

camera = cv2.VideoCapture(0)
check_settings()
VIDEO = VideoStreaming()

def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

############################################## THE REAL DEAL ###############################################
@app.route('/detectObject' , methods=['POST'])
def mask_image():
	# print(request.files , file=sys.stderr)
	file = request.files['image'].read() ## byte file
	npimg = np.fromstring(file, np.uint8)
	img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
	######### Do preprocessing here ################
	# img[img > 150] = 0
	## any random stuff do here
	################################################

	img = runModel(img)

	img = Image.fromarray(img.astype("uint8"))
	rawBytes = io.BytesIO()
	img.save(rawBytes, "JPEG")
	rawBytes.seek(0)
	img_base64 = base64.b64encode(rawBytes.read())
	return jsonify({'status':str(img_base64)})

##################################################### THE REAL DEAL HAPPENS ABOVE ######################################

@app.route('/test' , methods=['GET','POST'])
def test():
	print("log: got at test" , file=sys.stderr)
	return jsonify({'status':'succces'})

@app.route('/')
def home():
	return render_template('./index.html')

@app.route('/live_video')
def live_video():
	return render_template('video.html')

@app.route('/video_feed')
def video_feed():
    '''
    Video streaming route.
    '''
    return Response(
        VIDEO.show(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/second')
def second():
	return render_template('./second.html')

@app.route('/practise')
def practise():
	return render_template('./practise.html')

@app.route('/real_time')
def real_time():
	return render_template('./real_time.html')

@app.after_request
def after_request(response):
    print("log: setting cors" , file = sys.stderr)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

# Button requests called from ajax
@app.route('/request_preview_switch')
def request_preview_switch():
    VIDEO.preview = not VIDEO.preview
    print('*'*10, VIDEO.preview)
    return "nothing"

@app.route('/request_flipH_switch')
def request_flipH_switch():
    VIDEO.flipH = not VIDEO.flipH
    print('*'*10, VIDEO.flipH)
    return "nothing"

@app.route('/request_model_switch')
def request_model_switch():
    VIDEO.detect = not VIDEO.detect
    print('*'*10, VIDEO.detect)
    return "nothing"

@app.route('/request_exposure_down')
def request_exposure_down():
    VIDEO.exposure -= 1
    print('*'*10, VIDEO.exposure)
    return "nothing"

@app.route('/request_exposure_up')
def request_exposure_up():
    VIDEO.exposure += 1
    print('*'*10, VIDEO.exposure)
    return "nothing"

@app.route('/request_contrast_down')
def request_contrast_down():
    VIDEO.contrast -= 4
    print('*'*10, VIDEO.contrast)
    return "nothing"

@app.route('/request_contrast_up')
def request_contrast_up():
    VIDEO.contrast += 4
    print('*'*10, VIDEO.contrast)
    return "nothing"

@app.route('/reset_camera')
def reset_camera():
    STATUS = reset_settings()
    print('*'*10, STATUS)
    return "nothing"

if __name__ == '__main__':
	app.run(debug = True)
