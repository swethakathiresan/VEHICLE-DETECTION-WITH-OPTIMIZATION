import cv2
import numpy as np
from time import sleep

largura_min = 80  # Largura minima do retangulo
altura_min = 80  # Altura minima do retangulo

offset = 6  # Erro permitido entre pixel

pos_linha = 550  # Posição da linha de contagem

delay = 60  # FPS do vídeo

detec = []
carros = 0
car_speeds = {}  # Dictionary to store speeds of each vehicle

# Calibration values (replace with your measured values)
pixels_per_meter = 10  # Adjust this based on your calibration
frames_per_second = 30  # Adjust this based on your video's frame rate

def pega_centro(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Check if network loaded successfully
if net.empty():
    print("Failed to load YOLO network.")
    exit()

classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()

# Check if layer names are empty
if not layer_names:
    print("No layers found in the YOLO network.")
    exit()

output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Check if output layers are empty
if not output_layers:
    print("No output layers found in the YOLO network.")
    exit()

cap = cv2.VideoCapture('video.mp4')
subtracao = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret, frame1 = cap.read()
    tempo = float(1 / delay)
    sleep(tempo)
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    img_sub = subtracao.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)

    height, width, channels = frame1.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame1, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == 'car':  # Change label to what you want to count
                # Draw rectangle around the car
                cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Count cars that pass the line
                centro = pega_centro(x, y, w, h)
                detec.append(centro)
                cv2.circle(frame1, centro, 4, (0, 0, 255), -1)
                for (cx, cy) in detec:
                    if cy < (pos_linha + offset) and cy > (pos_linha - offset):
                        carros += 1

                        # Calculate speed (calibrated speed calculation)
                        if carros not in car_speeds:
                            car_speeds[carros] = 0
                        car_speeds[carros] += pixels_per_meter / frames_per_second  # Adjust based on calibration

                        # Display car detection and speed information
                        print(f"Car {carros} is detected. Speed: {car_speeds[carros]:.2f} m/s")

                        # Display speed on top of the detection box
                        cv2.putText(frame1, f"Speed: {car_speeds[carros]:.2f} m/s", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                        detec.remove((cx, cy))

    cv2.putText(frame1, "VEHICLE COUNT : " + str(carros), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("Video Original", frame1)
    cv2.imshow("Detectar", dilatada)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()





# import cv2
# import numpy as np
# from time import sleep

# largura_min = 80  # Largura minima do retangulo
# altura_min = 80  # Altura minima do retangulo

# offset = 6  # Erro permitido entre pixel

# pos_linha = 550  # Posição da linha de contagem

# delay = 60  # FPS do vídeo

# detec = []
# carros = 0
# car_speeds = {}  # Dictionary to store speeds of each vehicle

# def pega_centro(x, y, w, h):
#     x1 = int(w / 2)
#     y1 = int(h / 2)
#     cx = x + x1
#     cy = y + y1
#     return cx, cy

# cap = cv2.VideoCapture('video.mp4')
# subtracao = cv2.bgsegm.createBackgroundSubtractorMOG()

# # Calibration values (replace with your measured values)
# pixels_per_meter = 10  # Adjust this based on your calibration
# frames_per_second = 30  # Adjust this based on your video's frame rate

# while True:
#     ret, frame1 = cap.read()
#     tempo = float(1 / delay)
#     sleep(tempo)
#     grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(grey, (3, 3), 5)
#     img_sub = subtracao.apply(blur)
#     dilat = cv2.dilate(img_sub, np.ones((5, 5)))
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
#     dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
#     contorno, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     cv2.line(frame1, (25, pos_linha), (1200, pos_linha), (255, 127, 0), 3)
#     for (i, c) in enumerate(contorno):
#         (x, y, w, h) = cv2.boundingRect(c)
#         validar_contorno = (w >= largura_min) and (h >= altura_min)
#         if not validar_contorno:
#             continue

#         cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         centro = pega_centro(x, y, w, h)
#         detec.append(centro)
#         cv2.circle(frame1, centro, 4, (0, 0, 255), -1)

#         for (cx, cy) in detec:
#             if cy < (pos_linha + offset) and cy > (pos_linha - offset):
#                 carros += 1

#                 # Calculate speed (calibrated speed calculation)
#                 if carros not in car_speeds:
#                     car_speeds[carros] = 0
#                 car_speeds[carros] += pixels_per_meter / frames_per_second  # Adjust based on calibration

#                 # Display car detection and speed information
#                 print(f"Car {carros} is detected. Speed: {car_speeds[carros]:.2f} m/s")

#                 # Display speed on top of the detection box
#                 cv2.putText(frame1, f"Speed: {car_speeds[carros]:.2f} m/s", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

#                 detec.remove((cx, cy))  

#     cv2.putText(frame1, "VEHICLE COUNT : " + str(carros), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
#     cv2.imshow("Video Original", frame1)
#     cv2.imshow("Detectar", dilatada)

#     if cv2.waitKey(1) == 27:
#         break

# cv2.destroyAllWindows()
# cap.release()
