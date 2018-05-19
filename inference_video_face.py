#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import sys
import time
import os
import numpy as np
import tensorflow as tf
import cv2
import argparse
import face_recognition
import _thread

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils_color as vis_util


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './protos/face_label_map.pbtxt'

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

###
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)

def run_yolo(out_filename):
    out = None

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    with detection_graph.as_default():
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      with tf.Session(graph=detection_graph, config=config) as sess:
        frame_num = 1490;
        while frame_num:
          frame_num -= 1
          ret, image = cap.read()
          if ret == 0:
              break

          if out is None:
              [h, w] = image.shape[:2]
              out = cv2.VideoWriter(out_filename, 0, 25.0, (w, h))

          # if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 10 == 0:
          if ret == True:
              image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
              # result image with boxes and labels on it.]
              image_np_expanded = np.expand_dims(image_np, axis=0)
              image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
              # Each box represents a part of the image where a particular object was detected.
              boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
              scores = detection_graph.get_tensor_by_name('detection_scores:0')
              classes = detection_graph.get_tensor_by_name('detection_classes:0')
              num_detections = detection_graph.get_tensor_by_name('num_detections:0')
              # Actual detection.
              start_time = time.time()
              (boxes, scores, classes, num_detections) = sess.run(
                  [boxes, scores, classes, num_detections],
                  feed_dict={image_tensor: image_np_expanded})
              elapsed_time = time.time() - start_time
              sys.stdout.write('Inference Time Cost: %s\r' % (format(elapsed_time)))
              sys.stdout.flush()
              # Do a gamma correction for darker images
              # pass the gamma corrected image frame
              #TODO Revisit this -- Face Rec not working on gamma correction
              # gamma = 1.5
              # adjusted = adjust_gamma(image, gamma=gamma)
              vis_util.visualize_boxes_and_labels_on_image_array(
                  image,
                  np.squeeze(boxes),
                  np.squeeze(classes).astype(np.int32),
                  np.squeeze(scores),
                  category_index,
                  known_face_encodings,
                  known_face_names,
                  use_normalized_coordinates=True,
                  line_thickness=4)
              out.write(image)

        cap.release()
        out.release()

def run():
    timestr = time.strftime("%Y%m%d-%H%M%S")
    video_file = "./media/detection_vid_" + timestr + ".avi"
    run_yolo(video_file)


#########
# FUNCTION DEFINITIONS END HERE
#########

#Init Face encoding data
known_face_encodings = []
known_face_names = []
source = './POI'

# List of allowed image extensions
valid_images = [".jpg",".png"]

### Get the sample Images and save face_encodings
for root, dirs, filenames in os.walk(source):
    for filename in filenames:
        ext = os.path.splitext(filename)[1]
        if ext.lower() not in valid_images:
            continue
        fullpath = os.path.join(source, filename)
        poi_image = face_recognition.load_image_file(fullpath)
        poi_image_face_encoding = face_recognition.face_encodings(poi_image)[0]
        known_face_encodings.append(poi_image_face_encoding)
        known_face_names.append(os.path.splitext(filename)[0])


## Set Options as per arguments
parser = argparse.ArgumentParser()
parser.add_argument("--mode", help="Video / Camera / IP")
parser.add_argument("--file_path", help="Optional : Video File Path. Required if mode = Video only")
parser.add_argument("--uname", help="username")
parser.add_argument("--secret", help="secret")
parser.add_argument("--addr", help="ip-address")
args = parser.parse_args()

if args.mode in ('Video', 'video'):
    if args.file_path:
        cap = cv2.VideoCapture(args.file_path)
elif args.mode in ('camera', 'Camera'):
    cap = cv2.VideoCapture(0)
elif args.mode in ('ip', 'IP'):
    if args.addr and args.uname and args.secret:
        url = 'rtsp://' + args.uname + ':' + args.secret + '@' + args.addr
        cap = cv2.VideoCapture(url)
else:
    print('Use -h for help')
    sys.exit()

#START
#TODO Run in multithread
run()
