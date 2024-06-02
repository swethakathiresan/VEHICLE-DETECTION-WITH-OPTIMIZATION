import cv2
import numpy as np
import time

# Define the objective function
def objective(confThreshold, nmsThreshold):
    global total_counters
    # Reset the vehicle count and other necessary variables
    vehicle_count = {class_name: 0 for class_name in classNames}
    detected_vehicles = set()
    total_time_start = time.time()
    
    # Initialize timing variables
    last_vehicle_time = time.time()  # Initialize last_vehicle_time
    
    # Process the video frame by frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Measure time taken to detect each vehicle
        vehicle_time_start = time.time()
        
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (widthHeight, widthHeight), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        layersNames = net.getLayerNames()
        output_names = [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

        outputs = net.forward(output_names)
        detected_class, detected_vehicle_count, last_vehicle_time = findObjects(outputs, frame, vehicle_count, last_vehicle_time)
        
        # Measure time taken to detect each vehicle
        vehicle_time_end = time.time()

        # Calculate time taken to detect each vehicle
        vehicle_time_taken = vehicle_time_end - vehicle_time_start
        print(f"Time taken to detect this vehicle(simulated annealing detection): {vehicle_time_taken} seconds")

        if detected_class:
            detected_vehicles.add(detected_class)
            total_counters += 1

        # Check if the total number of detected vehicles reaches 147
        if sum(vehicle_count.values()) >= 20:
            print("Total number of vehicles detected (simulated annealing detection): ", sum(vehicle_count.values()))
            break

    total_time_end = time.time()
    total_time_taken = total_time_end - total_time_start
    return total_time_taken

# Simulated Annealing algorithm
def simulated_annealing(objective, bounds, n_iterations, step_size, temp):
    # Generate an initial point
    best = np.array([confThreshold, nmsThreshold])
    best_eval = objective(best[0], best[1])
    curr, curr_eval = best, best_eval
    
    # Run the algorithm
    for i in range(n_iterations):
        # Take a step
        candidate = best + np.random.uniform(-step_size, step_size, 2)
        candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
        
        # Evaluate candidate point
        candidate_eval = objective(candidate[0], candidate[1])
        
        # Check for new best solution
        if candidate_eval < best_eval:
            best, best_eval = candidate, candidate_eval
    
    return best, best_eval

# Path to the video file
video_path = r"vehicle.mp4"

# Create a VideoCapture object
cap = cv2.VideoCapture(video_path)
total_counters = 147

# Simulated Annealing parameters
T = 1000  # Initial temperature
T_min = 0.00001  # Minimum temperature limit
alpha = 0.9  # Reducing rate

# Initial values for confThreshold and nmsThreshold
confThreshold = 0.5
nmsThreshold = 0.2

# Check if the video is opened successfully
if not cap.isOpened():
    print("Error: Unable to open the video file.")
else:
    print("Video file opened successfully.")

    # Load class names
    classFile = r'coco.names'
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    # Load YOLO model
    modelConfiguration = r'yolov3.cfg'
    modelWeights = r'yolov3.weights'
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Function to find objects in the frame
    def findObjects(outputs, frame, vehicle_count, last_vehicle_time):
        height, width, _ = frame.shape
        detected_vehicle_count = sum(vehicle_count.values()) + 1  # Start from 1
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    class_name = classNames[classId]
                    vehicle_count[class_name] += 1  # Update vehicle count for detected class
                    print(f"Vehicle count: {detected_vehicle_count}, Detected: {class_name} with confidence {confidence}. Total {class_name}s: {vehicle_count[class_name]}")
                    return class_name, detected_vehicle_count, time.time()  # Return class name, detected_vehicle_count, and updated time
        return None, detected_vehicle_count, last_vehicle_time

    # Parameters for object detection
    confThreshold = 0.5
    nmsThreshold = 0.2

    # Set the width and height of the input blob
    widthHeight = 320

    # Initialize vehicle count
    vehicle_count = {class_name: 0 for class_name in classNames}

    # Initialize variables to keep track of detected vehicles
    detected_vehicles = set()

    # Initialize timing variables
    total_time_start = time.time()
    last_vehicle_time = time.time()  # Initialize last_vehicle_time

    # Process the video frame by frame
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        # Measure time taken to detect each vehicle
        vehicle_time_start = time.time()
        
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (widthHeight, widthHeight), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        layersNames = net.getLayerNames()
        output_names = [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

        outputs = net.forward(output_names)
        detected_class, detected_vehicle_count, last_vehicle_time = findObjects(outputs, frame, vehicle_count, last_vehicle_time)
        
        # Measure time taken to detect each vehicle
        vehicle_time_end = time.time()

        # Calculate time taken to detect each vehicle
        vehicle_time_taken = vehicle_time_end - vehicle_time_start
        print(f"Time taken to detect this vehicle(yolo time taken): {vehicle_time_taken} seconds")
        last_vehicle_time = vehicle_time_end

        if detected_class:
            detected_vehicles.add(detected_class)
            total_counters += 1

        # Check if the total number of detected vehicles reaches 147
        if sum(vehicle_count.values()) >= 20:
            print("Total number of vehicles detected(yolo detected): ", sum(vehicle_count.values()))
            break

    # Calculate total time taken to detect all vehicles
    total_time_end = time.time()
    total_time_taken = total_time_end - total_time_start
    print(f"Total time taken to detect all vehicles(yolo printed): {total_time_taken} seconds")

    # Define range for input
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    # Perform the simulated annealing search
    best, score = simulated_annealing(objective, bounds, 1, 0.1, 1000)
    print(f"Done! f({best}) = {score}")
    cap.release()

