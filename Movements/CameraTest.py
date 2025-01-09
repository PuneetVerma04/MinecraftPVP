import cv2

# Replace with your IP Webcam URL
url = "http://192.168.137.139:8080/video"

# Open a connection to the video stream
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Error: Unable to connect to the IP Webcam")
    exit()

# Define a scaling factor (e.g., 50% of original size)
scaling_factor = 0.5  # Adjust this value as needed

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize the frame
    new_width = int(frame.shape[1] * scaling_factor)
    new_height = int(frame.shape[0] * scaling_factor)
    resized_frame = cv2.resize(frame, (new_width, new_height))

    # Display the resized frame
    cv2.imshow("IP Webcam (Resized)", resized_frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
