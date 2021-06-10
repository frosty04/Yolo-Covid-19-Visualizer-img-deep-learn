import numpy as np
import cv2
from scipy.spatial import distance as dist

SIZE = (320, 320)
MIN_CONFIDENCE = 0.5

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

def get_boxes(outputs, height, width): 
    boxes = []
    objects_id = []

    for output in outputs:
        for detect in output:
            scores = detect[5:] # per prendere solo i valori di probabile appartenenza alle classi
            class_id = np.argmax(scores) # funzione di num.py per restituire valore massimo, quindi
            # in questo caso la probabilità di appartenenza alle classi maggiore
            conf = scores[class_id]

            if(conf>MIN_CONFIDENCE):
                
                center_x = int(detect[0]*width) # "stretching" del box
                center_y = int(detect[1]*height) # "stretching" del box

                w = int(detect[2]*width) # larghezza
                h = int(detect[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append((x, y, w, h))
                objects_id.append(class_id)

    return objects_id, boxes


def draw_results(img, boxes, class_ids):
    for box, class_id, in zip(boxes, class_ids):
        x, y, w, h = box
        color = colors[class_id]
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2) # immagine, coordinate origine, "fine" box, colore, spessore rettangolo
        label = classes[class_id]

        cv2.putText(img, label, (x, y-5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color, 1) # "3" == coordinate
    return img



net, output_layers, classes, colors = load_yolo()
img = cv2.imread("bikes.jpg")
outputs, blob = detect_objects(img, net, output_layers)
objects_id, boxes = get_boxes(outputs, img.shape[0], img.shape[1])
img_out = draw_results(img, boxes, objects_id)

cv2.imshow("Output", img_out)
cv2.waitKey(0) # senza questo, loop infinito di apri e chiudi