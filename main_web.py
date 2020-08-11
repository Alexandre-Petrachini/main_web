#!/usr/bin/env python
#
# Project: Video Streaming with Flask
# Author: Log0 <im [dot] ckieric [at] gmail [dot] com>
# Date: 2014/12/21
# Website: http://www.chioka.in/
# Description:
# Modified to support streaming out with webcams, and not just raw JPEGs.
# Most of the code credits to Miguel Grinberg, except that I made a small tweak. Thanks!
# Credits: http://blog.miguelgrinberg.com/post/video-streaming-with-flask
#
# Usage:
# 1. Install Python dependencies: cv2, flask. (wish that pip install works like a charm)
# 2. Run "python main.py".
# 3. Navigate the browser to the local webpage.
from flask import Flask, render_template, Response, g, stream_with_context
from camera import VideoCamera

# Network Algorithm import
import posenet
import mobilenet
from posenet.posenet_lite_web import PoseNet
from mobilenet.mobilenet_lite import MobileNet
#from inception_v3.inception_v3_lite import Inceptionv3Net
from tinyyolo.tinyyolo_lite import TinyYoloNet
#from mtcnn.mtcnn import Mtcnn
from emotion.emotion_network import EmoNet
from emotion.emotion_result import detect_faces
from emotion.emotion_result import draw_text
from emotion.emotion_result import draw_bounding_box
from emotion.emotion_result import apply_offsets
from lpdr.src.network.license_plate_detection import LicensePlateDetection
# Python Library Import
import numpy as np
import time
import cv2

app = Flask(__name__)
# Global Config
mode = 0 # 0: LNE, 1: CPU
cam_id = 0 # for DQ1 usb
# Global Variable
model_init_lock = 0
model_select    = 0
answer_txt      = ''

# emotion parameter
face_detection = cv2.CascadeClassifier("../../models/haarcascade_frontalface_default.xml")
frame_window = 10
emotion_offsets = (20, 40)
emotion_window = []
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',4: 'sad', 5: 'surprise', 6: 'neutral'}
color = (0, 0, 255)
# emotion parameter end

@app.route('/')
def index():
    return render_template('index.html', image_file='image/CTO_Techcon.png')


@app.route('/led/<action>')
def turn_led_onoff(action):
    global model_init_lock
    global model_select

    # model_select
    # 0 : original video
    # 1 : Posenet
    # 2 : Mobilenet
    # 3 : Inception v3
    # 4 : TinyYolo
    # 5 : MTCNN
	# 6 : LPDR
    if (action == "posenet"):
        model_init_lock = 1
        model_select    = 1
        print("#############################################")
        print("local mode select :", model_select)
        print("Hello DQ1 Algorithm Studio-232! posenet  on!")
        print("#############################################")

    if (action == "mobilenet"):
        model_init_lock = 1
        model_select = 2
        print("#############################################")
        print("local mode select :", model_select)
        print("Hello DQ1 Algorithm Studio-232! mobilenet on!")
        print("#############################################")

    if (action == "inc3"):
        model_init_lock = 1
        model_select = 3
        print("################################################")
        print("local mode select :", model_select)
        print("Hello DQ1 Algorithm Studio-232! inception v3 on!")
        print("################################################")

    if (action == "tyolo"):
        model_init_lock = 1
        model_select = 4
        print("################################################")
        print("local mode select :", model_select)
        print("Hello DQ1 Algorithm Studio-232! Tiny-Yolo on!")
        print("################################################")

    if (action == "mtcnn"):
        model_init_lock = 1
        model_select = 5
        print("##################################################")
        print("local mode select :", model_select)
        print("Hello DQ1 Algorithm Studio-232! MTCNN image on!")
        print("##################################################")

    if (action == "emotion"):
        model_init_lock = 1
        model_select = 6
        print("##################################################")
        print("local mode select :", model_select)
        print("Hello DQ1 Algorithm Studio-232! emotion on!")
        print("##################################################")
    if (action == "lpdr"):
        model_init_lock = 1
        model_select = 7
        print("##################################################")
        print("local mode select :", model_select)
        print("##################################################")

    if (action == "orig"):
        model_init_lock = 1
        model_select = 0
        print("##################################################")
        print("local mode select :", model_select)
        print("Hello DQ1 Algorithm Studio-232! original image on!")
        print("##################################################")
    if(action == "next"):
        model_init_lock = 1
        model_select = model_select + 1
    if (model_select > 6):
        model_select = 0
        model_init_lock = 1
        print("##################################################")
        print("local mode select :", model_select)
        print("Hello DQ1 Algorithm Studio-232! Next on!")
        print("##################################################")

    return (''), 204


def gen(camera):
    skip_counter = 0

    global model_select
    global model_init_lock
    global answer_txt

    answer_bf = 'dumy'

#    Pnet = PoseNet(mode)
#    Mnet = MobileNet("./models/mobilenet.lne", "./mobilenet/labels.txt")
#    Inet = Inceptionv3Net("./models/inception_v3.lne", "./inception_v3/labels.txt")
#    Tnet = TinyYoloNet("./models/tiny_yolo.lne")
#    Mtcnn_C = Mtcnn()
#Enet = EmoNet("./models/emotion.lne")
    lpdr_net = LicensePlateDetection('models/bin/baseline1/lpd_net.bin','models/bin/baseline1/lpd_net_data.bin','models/bin/baseline1/lpd.meta')

    while True:
        s0 = time.time()
        orig_img = camera.get_frame()


        # Original video stream and Network Init Sequence
        if(model_init_lock == 1 or model_select == 0):
            print("orig model select :", model_select)
            orig_img = cv2.flip(orig_img, 1)
            frame = cv2.imencode('.bmp', orig_img)[1].tobytes()

            if (model_select == 1):
                model_init_lock = 0
            elif (model_select == 2):
                model_init_lock = 0
            elif (model_select == 3):
                model_init_lock = 0
            elif (model_select == 4):
                model_init_lock = 0
            elif (model_select == 5):
                model_init_lock = 0
            elif (model_select == 6):
                model_init_lock = 0
            elif (model_select == 7):
                model_init_lock = 0
            else :
                model_init_lock = 0
                model_select    = 0

        # PoseNet run, if event occur
        elif(model_init_lock == 0 and model_select == 1):
#            print("posenet model select :", model_select)
            if orig_img is None or skip_counter%3 != 2:
                print("not get Video frame")
                skip_counter += 1
                continue
            orig_img                   = Pnet.crop_image(orig_img)
            (factor, input_img)        = Pnet.resize_img(orig_img)
            (heat, offset, dfwd, dbwd) = Pnet.inference(input_img)
            poses                      = Pnet.post_process(heat, offset, dfwd, dbwd, factor)
            orig_img                   = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            frame                      = Pnet.post_draw(orig_img, poses)
            s5 = time.time()
#            print("Total :", s5 - s0)

		# LPDR run, if event occur
        elif(model_init_lock == 0 and model_select == 7):
			#print("LPDR model select :", model_select)
            persist = True
            stg2_img = np.zeros((1080,1920,3), dtype=np.uint8)
            stg2_bbox = [0,0,0,0]
            stg1_total_time, stg2_total_time = [], []
            img = cv2.imread(orig_img, cv2.IMREAD_COLOR)
            stg1_img, stg1_bbox, stg1_times = lp_detection(args, net, img)
            stg1_total_time += [stg1_times]
            if stg1_img.size and stg1_img.shape[0] != 0 and stg1_img.shape[1] != 0:
                stg2_img, stg2_bbox, stg2_times = lp_detection(args, net, stg1_img, stg1_bbox)
                stg2_total_time += [stg2_times]
            else:
                stg2_img = img
                stg2_bbox = [0,0,0,0]
                print('-- Car Not Found')
                output_file = os.path.join(output_path, "res_{}.txt".format(os.path.splitext(os.path.basename(fname))[0]))
                np.savetxt(output_file, [stg2_bbox], fmt="%d,%d,%d,%d")

            if persist:
                output_cropped = os.path.join(args.output_path, "cropped", os.path.relpath(fname, args.dataset_path))
                safe_create_dir(os.path.dirname(output_cropped))
                cv2.imwrite(output_cropped, stg2_img)

        # MobileNet run, if event occur
        elif(model_init_lock == 0 and model_select == 2):
#            print("mobilenet model select :", model_select)
            input_img = Mnet.crop_image(orig_img)
            input_img = Mnet.resize_img(input_img)
            answer    = Mnet.inference(input_img)

            frame     = Mnet.post_draw(orig_img, answer)

            if (answer != answer_bf):
                answer_txt = Mnet.answer_label(answer)
                f_answer = open("answer.txt", 'w')
                f_answer.write(Mnet.answer_label(answer))
                f_answer.close()

            answer_bf = answer
            s5 = time.time()
#            print("Total :", s5 - s0)

        # Inceptionv3 run, if event occur
        elif(model_init_lock == 0 and model_select == 3):
#            print("inceptionv3 model select :", model_select)
            input_img = Inet.crop_image(orig_img)
            input_img = Inet.resize_img(input_img)
            answer    = Inet.inference(input_img)

            frame     = Inet.post_draw(orig_img, answer)

            if (answer != answer_bf):
                f_answer = open("answer.txt", 'w')
                f_answer.write(Inet.answer_label(answer))
                f_answer.close()

            answer_bf = answer
            s5 = time.time()
#            print("Total :", s5 - s0)

        # Tiny yolo run if event occur
        elif(model_init_lock == 0 and model_select == 4):
#            print("Tiny yolo model select :", model_select)
            orig_img = cv2.flip(orig_img, 1)
            input_img_1     = Tnet.crop_image(orig_img)
            input_img_2     = Tnet.resize_img(input_img_1)
            lne_result    = Tnet.inference(input_img_2)
            results       = Tnet.post_process(lne_result[0,0,0,:])
#            frame         = Tnet.post_draw(input_img, results)
            frame         = Tnet.post_draw(input_img_1, results)

            s5 = time.time()
#            print("Total :", s5 - s0)

        # MTCNN run if event occur
        elif(model_init_lock == 0 and model_select == 5):
#            print("MTCNN model select :", model_select)
            orig_img = cv2.flip(orig_img, 1)
 #           input_img_1     = Mtcnn_C.crop_image(orig_img)
 #           input_img_2     = Mtcnn_C.resize_img(input_img_1)

 #           frame = cv2.imencode('.bmp', orig_img)[1].tobytes()

  #          lne_result    = Tnet.inference(input_img_2)
  #         input_img_1     = Mtcnn_C.resize_img(orig_img)
            frame       = Mtcnn_C.inference(orig_img)
  #          frame         = Mtcnn_C.post_draw(input_img_1, results)

            s5 = time.time()
#            print("Total :", s5 - s0)

        elif(model_init_lock == 0 and model_select == 6):
#            print("Emotion model select :", model_select)

            orig_img = cv2.flip(orig_img, 1)
            gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
            rgb_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            faces = detect_faces(face_detection, gray_img)
            for face_coordinates in faces:
                x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
                if x1 <0 or x2 <0 or y1 <0 or y2<0:
                    continue
                gray_face = gray_img[y1:y2, x1:x2]
                input_img = Enet.resize_img(gray_face)
                answer, prob = Enet.inference(input_img)
                emotion_text = emotion_labels[answer]
                emotion_text = emotion_text +": "+ str(prob.round()) + "%"
                print(emotion_text)
                draw_bounding_box(face_coordinates, rgb_img, color)
                draw_text(face_coordinates, rgb_img, emotion_text, color, 0, -20, 1, 2)
            frame = Enet.post_draw(rgb_img)

            s5 = time.time()
#            print("Total :", s5 - s0)



        else:
            model_init_lock = 0
            model_select = 0


        yield (b'--frame\r\n'
               b'Content-Type: image/bmp\r\n\r\n' + frame + b'\r\n\r\n')
        skip_counter = 0


#def gen_txt():
#    global answer_txt
    #answer_txt = "hello world"
#    yield answer_txt


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera(cam_id)), mimetype='multipart/x-mixed-replace; boundary=frame')

#@app.route('/txt_feed')
#def txt_feed():
#    return Response(stream_with_context(gen_txt()))

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
