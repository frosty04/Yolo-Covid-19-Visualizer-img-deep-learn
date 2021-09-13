import numpy as np
import cv2
from numpy.core.numeric import indices
from scipy.spatial import distance as dist
from scipy.spatial.kdtree import distance_matrix
import os.path

SIZE = (320, 320)
MIN_CONFIDENCE = 0.5
NMS_THRESHOLD = 0.2 # "sensibilità" del NMS

TARGET_CLASS = ["person"] # così da poter visualizzare solo le persone

MIN_DISTANCE = 30  # 30 pixel = 1 metro

RED = (0, 0, 255)
GREEN = (0, 255, 0)

LINE_THICKNESS = 1
CIRCLE_RADIUS = 2

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

def compute_distances(centroids):
    dist_matrix = dist.cdist(centroids, centroids) # dist.cdist = "misura/disatanza tra i 2 punti"
    dist_matrix = dist_matrix+np.eye(dist_matrix.shape[0], dist_matrix.shape[1])*1000 # np.eye = "misura/distanza diagonale"
    return dist_matrix

def get_contact_indices(centroids):
    dist_matrix = compute_distances(centroids)
    indices = np.where(dist_matrix<MIN_DISTANCE) # ritorna indici "dove questa condizione è verificata"
    contact_indices = list(zip(indices[0], indices[1]))
    return contact_indices
    # funzione usata per i "falsi positivi"

def non_max_suppression(boxes, centroids, confidences, min_confidence, threshold):

    boxesMax = []
    centroidsMax = []
    boxesIds = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, threshold)

    for boxId in boxesIds:
        boxesMax.append(boxes[boxId[0]])
        centroidsMax.append(centroids[boxId[0]])

    return boxesMax, centroidsMax



def draw_results(img, centroids, contact_indices):

    centroids_drawn = set()

    for c1, c2 in contact_indices:
        centroid1 = centroids[c1]
        centroid2 = centroids[c2]

        cv2.circle(img, centroid1, CIRCLE_RADIUS, RED, cv2.FILLED) # rosso
        cv2.circle(img, centroid2, CIRCLE_RADIUS, GREEN, cv2.FILLED) # rosso

        cv2.line(img, centroid1, centroid2, RED, thickness=LINE_THICKNESS) # linea tra i "centroidi"

        centroids_drawn.add(centroid1)
        centroids_drawn.add(centroid2)

    centroids_to_draw = set(centroids) - centroids_drawn

    for centroid in centroids_to_draw:
        cv2.circle(img, centroid, CIRCLE_RADIUS, GREEN, cv2.FILLED) # "(0, 255, 0)" == verde
        
    return img

input_video_path = input("Inseririre il percorso del video: ")    

if(not os.path.isfile(input_video_path)):
    print("File " + input_video_path +" non trovato")
    exit(0)

cap = cv2.VideoCapture(input_video_path)
ret, frame = cap.read() # "ret" == boolean

if (not ret):
    print("Errore durante il caricamento del video")
    exit(0)

print("File " + input_video_path + " caricato con succcesso")

input_video_name, input_video_ext = input_video_path.split("/")[-1].split(".") # prima parte == nome, seconda == estensione
output_video_path = input_video_name+"_output."+input_video_ext

codec = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, codec, 20.0, (frame.shape[1], frame.shape[0])) #20.0 == frame
print("Il video verrà salvato come" + output_video_path)

frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Numero frame: %d" % frames_count)
current_frame = 1

net, output_layers, classes, colors = load_yolo()

print("Modello YOLO caricato !")
print("Inizio analisi video")

while(cap.isOpened()):

    ret, frame = cap.read()

    if (not ret):
        break

    outputs, blob = detect_objects(frame, net, output_layers)
    boxes, centroids, confidences = get_detection_metadata(outputs, frame.shape[0], frame.shape[1])
    _, centroids = non_max_suppression(boxes, centroids, confidences, MIN_CONFIDENCE, NMS_THRESHOLD) # nome var == "_" vuol dire che sono variabili di cui il valore non è più usato 

#print(cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, 0.3)) # box che rimangono dopo la "non maximum suppression"

    contact_indices = get_contact_indices(centroids)
    img_out = draw_results(frame, centroids, contact_indices)

    cv2.imshow("Output with NMS", img_out)
    out.write(img_out)

    print("Frame %d di %d analizzato" % (current_frame, frames_count))
    current_frame+=1

    #cv2.waitKey(0) # senza questo, loop infinito di apri e chiudi, usato per le immagini
    if(cv2.waitKey(1)==ord("q")): # usato per i video, scorre i frame automaticamente
        print("Analisi del video interrotta!")
        break

cap.release()
out.release()
cv2.destroyAllWindows()