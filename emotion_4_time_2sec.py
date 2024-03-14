import cv2
from deepface import DeepFace
from openpyxl import Workbook
from datetime import datetime
import time

# Load the pre-trained emotion detection model
model = DeepFace.build_model("Emotion")

# Define emotion labels
emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start capturing video
cap = cv2.VideoCapture(0)

# Create a new Excel workbook and select the active sheet
workbook = Workbook()
sheet = workbook.active

# Write headers to the Excel sheet
sheet["A1"] = "Time"
sheet["B1"] = "Emotion"
sheet["C1"] = "Confidence"

row = 2  # Starting row for data in Excel

start_time = time.time()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # Get current time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for x, y, w, h in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = gray_frame[y : y + h, x : x + w]

        # Resize the face ROI to match the input shape of the model
        resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)

        # Normalize the resized face image
        normalized_face = resized_face / 255.0

        # Reshape the image to match the input shape of the model
        reshaped_face = normalized_face.reshape(1, 48, 48, 1)

        # Predict emotions using the pre-trained model
        preds = model.predict(reshaped_face)[0]
        emotion_idx = preds.argmax()
        emotion = emotion_labels[emotion_idx]
        confidence = preds[emotion_idx]

        # Draw rectangle around face and label with predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(
            frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2
        )

        # Write time, emotion, and confidence to Excel sheet every 2 seconds
        elapsed_time = time.time() - start_time
        if elapsed_time >= 2:
            sheet[f"A{row}"] = current_time
            sheet[f"B{row}"] = emotion
            sheet[f"C{row}"] = confidence
            row += 1
            start_time = time.time()

    # Display the resulting frame
    cv2.imshow("Real-time Emotion Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Save the Excel workbook
workbook.save("emotion_results_time_2sec.xlsx")

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
