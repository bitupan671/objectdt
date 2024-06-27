import tkinter as tk
import cv2

flag = True
cap = None  # Global variable to hold the capture object

# Function to detect objects in the image
def detect_objects():
    global flag, cap
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    className = []
    classFile = 'coco.names'
    with open(classFile, 'rt') as f:
        className = f.read().rstrip('\n').split('\n')

    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weigthsPath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weigthsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    while flag:
        success, img = cap.read()
        classIds, confs, bbox = net.detect(img, confThreshold=0.5)
        print(classIds, bbox)

        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, className[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Output", img)
        if cv2.waitKey(1) == ord('q'):  # Press 'q' to exit
            break

    # Release the camera capture object when loop ends or 'q' is pressed
    if cap is not None:
        cap.release()

# Create main window
window = tk.Tk()
window.title("Object Detection App")
window.geometry('1280x760')

# Create buttons
stop_button = tk.Button(window, text="STOP", command=window.quit)
stop_button.pack(side=tk.LEFT, padx=5, pady=5)

detect_button = tk.Button(window, text="Detect Objects", command=detect_objects)
detect_button.pack(side=tk.RIGHT, padx=5, pady=5)

# Run the Tkinter event loop
window.mainloop()
