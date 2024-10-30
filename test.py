import cv2
import time
from ultralytics import YOLO
from threading import Thread, Lock
from queue import Queue, PriorityQueue

# Load the exported OpenVINO model
ov_model = YOLO("./yolov8n_openvino_model/")

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Shared variables
frame_queue = Queue()
processed_queue = PriorityQueue()  # Holds frames in order based on their sequence
fps_lock = Lock()
fps = 0
prev_time = time.time()
running = True

def capture_frames():
    """Capture frames from the webcam and add them to the frame queue with a sequence number."""
    sequence = 0
    while running:
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put((sequence, frame))
        sequence += 1

def process_frames():
    """Process frames and perform inference using YOLO."""
    global fps, prev_time
    while running:
        if not frame_queue.empty():
            # Get the next frame from the queue
            sequence, frame = frame_queue.get()
            if frame is None:
                continue
            
            # Run inference
            results = ov_model(frame)
            annotated_frame = results[0].plot()

            # Calculate FPS
            with fps_lock:
                current_time = time.time()
                fps = 1 / (current_time - prev_time)
                prev_time = current_time

            # Display FPS on the frame
            fps_text = f"FPS: {fps:.2f}"
            cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Add the processed frame to the output queue in order
            processed_queue.put((sequence, annotated_frame))

def display_frames():
    """Display processed frames in the correct sequence order."""
    next_sequence = 0
    while running:
        if not processed_queue.empty():
            sequence, frame = processed_queue.queue[0]  # Peek at the top item

            if sequence == next_sequence:
                # If the frame is in order, display it
                processed_queue.get()  # Remove the frame from the queue
                cv2.imshow("YOLOv8 OpenVINO Inference", frame)
                next_sequence += 1  # Update to the next expected sequence number

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Start threads
capture_thread = Thread(target=capture_frames)
processing_threads = [Thread(target=process_frames) for _ in range(2)]  # Multiple processing threads
display_thread = Thread(target=display_frames)

capture_thread.start()
for thread in processing_threads:
    thread.start()
display_thread.start()

# Wait for threads to complete
capture_thread.join()
for thread in processing_threads:
    thread.join()
display_thread.join()

# Release resources
running = False
cap.release()
cv2.destroyAllWindows()
