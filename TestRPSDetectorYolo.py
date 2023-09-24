import argparse
import cv2
from keras.models import model_from_json
import numpy as np

from yolo import YOLO

RPS_dict = {0: "Paper", 1: "Rock", 2: "Scissors"}


# load json and create model
json_file = open('RPSmodel/RPS_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
RPS_model = model_from_json(loaded_model_json)

# load weights into new model
RPS_model.load_weights("RPSmodel/RPS_model.h5")
print("Loaded model from disk")

ap = argparse.ArgumentParser()
ap.add_argument('-n', '--network', default="normal", choices=["normal", "tiny", "prn", "v4-tiny"],
                help='Network Type')
ap.add_argument('-d', '--device', type=int, default=0, help='Device to use')
ap.add_argument('-s', '--size', default=416, help='Size for yolo')
ap.add_argument('-c', '--confidence', default=0.2, help='Confidence for yolo')
ap.add_argument('-nh', '--hands', default=-1, help='Total number of hands to be detected per frame (-1 for all)')
args = ap.parse_args()

if args.network == "normal":
    print("loading yolo...")
    yolo = YOLO("yolo_models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
elif args.network == "prn":
    print("loading yolo-tiny-prn...")
    yolo = YOLO("yolo_models/cross-hands-tiny-prn.cfg", "models/cross-hands-tiny-prn.weights", ["hand"])
elif args.network == "v4-tiny":
    print("loading yolov4-tiny-prn...")
    yolo = YOLO("yolo_models/cross-hands-yolov4-tiny.cfg", "models/cross-hands-yolov4-tiny.weights", ["hand"])
else:
    print("loading yolo-tiny...")
    yolo = YOLO("yolo_models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])

yolo.size = int(args.size)
yolo.confidence = float(args.confidence)

print("starting webcam...")
cv2.namedWindow("preview")
vc = cv2.VideoCapture(args.device)

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
    frame = cv2.resize(frame, (1280, 720))
else:
    rval = False

while rval:
    width, height, inference_time, results = yolo.inference(frame)

    # display fps
    cv2.putText(frame, f'{round(1/inference_time,2)} FPS', (15,15), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,255,255), 2)

    # sort by confidence
    results.sort(key=lambda x: x[2])

    # how many hands should be shown
    hand_count = len(results)
    if args.hands != -1:
        hand_count = int(args.hands)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # display hands
    for detection in results[:hand_count]:
        id, name, confidence, x, y, w, h = detection
        cx = x + (w / 2)
        cy = y + (h / 2)

        # draw a bounding box rectangle and label on the image
        color = (0, 255, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict the RPSs
        RPS_prediction = RPS_model.predict(cropped_img)
        maxindex = int(np.argmax(RPS_prediction))
        cv2.putText(frame, RPS_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        
        
#         text = "%s (%s)" % (name, round(confidence, 2))
#         cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5, color, 2)

    cv2.imshow("RPS Detection", frame)

    rval, frame = vc.read()

    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

cv2.destroyWindow("preview")
vc.release()
