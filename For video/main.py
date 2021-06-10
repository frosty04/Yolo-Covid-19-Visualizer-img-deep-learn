import numpy as np
import cv2
from scipy.spatial import distance as dist

SIZE = (320, 320)
MIN_CONFIDENCE = 0.5
NMS_THRESHOLD = 0.2 # "sensibilità" del NMS

TARGET_CLASS = ["person", "bicycle"] # così da poter visualizzare solo le persone

def load_yolo(yolo_path=""):
    net = cv2.dnn.readNet(yolo_path+"yolov3.weights", yolo_path+"yolov3.cfg")
    classes = [] # lista vuota

    with open(yolo_path+"coco.names", "r") as f: #r = read
        classes = [line.strip() for line in f.readlines()] #minuto 13 video

    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()] # nodi "ultimo strato"

    colors = np.random.uniform(0, 255, size=(len(classes), 3)) #colori casuali, numero 3 per avere "l'rgb"

    return net, output_layers, classes, colors

def detect_objects(img, net, output_layers):
    img = cv2.resize(img, SIZE)
    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255, size = SIZE) #input per la rete neurale
    net.setInput(blob)
    outputs = net.forward(output_layers) # "calcolo" vero e proprio che la rete neurale fa
    return outputs, blob




#print(outputs[0], (outputs[0].shape))

#print(type(net), type(output_layers))
#print(colors.shape)
#print(classes)

def get_detection_metadata(outputs, height, width):

    boxes = []
    confidences = []
    centroids = []

    for output in outputs:
        for detect in output:
            scores = detect[5:] # per prendere solo i valori di probabile appartenenza alle classi
            class_id = np.argmax(scores) # funzione di num.py per restituire valore massimo, quindi
            # in questo caso la probabilità di appartenenza alle classi maggiore
            conf = scores[class_id]

            if(classes[class_id] in TARGET_CLASS and conf>MIN_CONFIDENCE):
                center_x = int(detect[0]*width) # "stretching" del box
                center_y = int(detect[1]*height) # "stretching" del box

                w = int(detect[2]*width) # larghezza
                h = int(detect[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append((x, y, w, h))
                confidences.append(conf.astype(float))
                centroids.append((center_x, center_y)) # ricordati doppie "(("

    return boxes, centroids, confidences


def non_max_suppression(boxes, confidences, min_confidence, threshold):

    boxesMax = []
    boxesIds = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, threshold)

    for boxId in boxesIds:
        boxesMax.append(boxes[boxId[0]])

    return boxesMax



def draw_results(img, centroids):

    for centroid in centroids:
        cv2.circle(img, centroid, 2, (0, 255, 0), cv2.FILLED) # "(0, 255, 0)" == verde

    return img



net, output_layers, classes, colors = load_yolo()

cap = cv2.VideoCapture("video.mp4")
ret, frame = cap.read() # "ret" == boolean

if (not ret):
    print("Errore durante il caricamento del video")
    exit(0)

while(cap.isOpened()):

    ret, frame = cap.read()

    if (not ret):
        print("Errore durante il caricamento del video")
        break

    outputs, blob = detect_objects(frame, net, output_layers)
    boxes, centroids, confidences = get_detection_metadata(outputs, frame.shape[0], frame.shape[1])
    boxes = non_max_suppression(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)
    print(boxes)

#print(cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, 0.3)) # box che rimangono dopo la "non maximum suppression"

    img_out = draw_results(frame, centroids)

    cv2.imshow("Output with NMS", img_out)
    #cv2.waitKey(0) # senza questo, loop infinito di apri e chiudi, usato per le immagini
    if(cv2.waitKey(1)==ord("q")): # usato per i video, scorre i frame automaticamente
        break

cap.release()
cv2.destroyAllWindows()