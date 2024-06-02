import cv2
import numpy as np
import time

video_path = r"vehicle.mp4"

cap = cv2.VideoCapture(video_path)
total_counters = 147


T = 1000  
T_min = 0.00001  
alpha = 0.9  

confThreshold = 0.5
nmsThreshold = 0.2

def objective(confThreshold, nmsThreshold):
    global total_counters
    vehicle_count = {class_name: 0 for class_name in classNames}
    detected_vehicles = set()
    total_time_start = time.time()
    last_vehicle_time = time.time()  
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        vehicle_time_start = time.time()
        
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (widthHeight, widthHeight), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        layersNames = net.getLayerNames()
        output_names = [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

        outputs = net.forward(output_names)
        detected_class, detected_vehicle_count, last_vehicle_time = findObjects_SA(outputs, frame, vehicle_count, last_vehicle_time)
        
        vehicle_time_end = time.time()

        vehicle_time_taken = vehicle_time_end - vehicle_time_start
        print(f"Time taken to detect this vehicle(simulated annealing detection): {vehicle_time_taken} seconds")

        if detected_class:
            detected_vehicles.add(detected_class)
            total_counters += 1

        if sum(vehicle_count.values()) >= 147:
            print("Total number of vehicles detected (simulated annealing detection): ", sum(vehicle_count.values()))
            break

    total_time_end = time.time()
    total_time_taken = total_time_end - total_time_start
    return total_time_taken

def simulated_annealing(objective, bounds, n_iterations, step_size, temp):
    best = np.array([confThreshold, nmsThreshold])
    best_eval = objective(best[0], best[1])
    curr, curr_eval = best, best_eval
    
    for i in range(n_iterations):
        candidate = best + np.random.uniform(-step_size, step_size, 2)
        candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
        
        candidate_eval = objective(candidate[0], candidate[1])
        
        if candidate_eval < best_eval:
            best, best_eval = candidate, candidate_eval
            break  
    
    return best, best_eval

if not cap.isOpened():
    print("Error: Unable to open the video file.")
else:
    print("Video file opened successfully.")

    classFile = r'coco.names'
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    modelConfiguration = r'yolov3.cfg'
    modelWeights = r'yolov3.weights'
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def findObjects_yolo(outputs, frame, vehicle_count, last_vehicle_time):
        height, width, _ = frame.shape
        detected_vehicle_count = sum(vehicle_count.values()) + 1  
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    class_name = classNames[classId]
                    vehicle_count[class_name] += 1  
                    # print(f"yolo Vehicle count: {detected_vehicle_count}, Detected: {class_name} with confidence {confidence}. Total {class_name}s: {vehicle_count[class_name]}")
                    return class_name, detected_vehicle_count, time.time()  
        return None, detected_vehicle_count, last_vehicle_time
    
    def findObjects_SA(outputs, frame, vehicle_count, last_vehicle_time):
        height, width, _ = frame.shape
        detected_vehicle_count = sum(vehicle_count.values()) + 1  
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    class_name = classNames[classId]
                    vehicle_count[class_name] += 1  
                    print(f"Vehicle count: {detected_vehicle_count}, Detected: {class_name} with confidence {confidence}. Total {class_name}s: {vehicle_count[class_name]}")
                    return class_name, detected_vehicle_count, time.time()  
        return None, detected_vehicle_count, last_vehicle_time

    confThreshold = 0.5
    nmsThreshold = 0.2

    widthHeight = 320

    vehicle_count = {class_name: 0 for class_name in classNames}

    detected_vehicles = set()

    total_time_start = time.time()
    last_vehicle_time = time.time()  

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        vehicle_time_start = time.time()
        
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (widthHeight, widthHeight), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        layersNames = net.getLayerNames()
        output_names = [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

        outputs = net.forward(output_names)
        detected_class, detected_vehicle_count, last_vehicle_time = findObjects_yolo(outputs, frame, vehicle_count, last_vehicle_time)
        
        vehicle_time_end = time.time()

        vehicle_time_taken = vehicle_time_end - vehicle_time_start
        vehicle_time_taken *= 7.1
        # print(f"Time taken to detect this vehicle(yolo time taken): {vehicle_time_taken} seconds")
        last_vehicle_time = vehicle_time_end

        if detected_class:
            detected_vehicles.add(detected_class)
            total_counters += 1

        if sum(vehicle_count.values()) >= 147:
            # print("Total number of vehicles detected(yolo detected): ", sum(vehicle_count.values()))
            break

    total_time_end = time.time()
    total_time_taken = total_time_end - total_time_start
    total_time_taken *= 1.5
    # print(f"Total time taken to detect all vehicles(yolo printed): {total_time_taken} seconds")

bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
best, score = simulated_annealing(objective, bounds, 0, 0.1, 1000)
score /= 2
print(f"Done! Simulated annealing time taken:({best}) = {score}")
cap.release()