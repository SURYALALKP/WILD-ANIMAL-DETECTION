from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import os
import sqlite3
from ultralytics import YOLO
from twilio.rest import Client
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
import time

# Add at the top
is_camera_on = False
camera = None
app = Flask(__name__)

# Constants
UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Load YOLO model
model = YOLO("runs/detect/train/weights/best.pt")

# Twilio credentials from environment variables
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
RECIPIENT_PHONE_NUMBER = os.getenv("RECIPIENT_PHONE_NUMBER")

uploaded_img_path = os.path.join(app.config["UPLOAD_FOLDER"], "uploaded.jpg")
detected_img_path = os.path.join("static/detected", "detected.jpg")
# similarly for videos

# Helper Functions
def allowed_video(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

def send_alert():
    alert_message = "🚨 Emergency! Elephant detected!"
    client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
    client.messages.create(
        body=alert_message,
        from_=TWILIO_PHONE_NUMBER,
        to=RECIPIENT_PHONE_NUMBER
    )
    
    # Log alert in database
    conn = sqlite3.connect("detections.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO alerts (message) VALUES (?)", (alert_message,))
    conn.commit()
    conn.close()

def init_db():
    conn = sqlite3.connect("detections.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS alerts (id INTEGER PRIMARY KEY AUTOINCREMENT, message TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
    cursor.execute("CREATE TABLE IF NOT EXISTS logs (id INTEGER PRIMARY KEY AUTOINCREMENT, animal TEXT, confidence REAL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
    conn.commit()
    conn.close()

def detect_objects(frame):
    results = model(frame)
    conn = sqlite3.connect("detections.db")
    cursor = conn.cursor()
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = "Elephant" if cls == 0 else f"Animal {cls}"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display label and confidence score only if confidence is greater than 0.8 (80%)
            if conf > 0.8:
                cv2.putText(frame, f"{label} {conf*100:.2f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # For lower confidence, only show bounding box without the label
                cv2.putText(frame, f"{conf*100:.2f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Log detection to database
            cursor.execute("INSERT INTO logs (animal, confidence) VALUES (?, ?)", (label, conf))
            conn.commit()

            #Send alert for Elephant detection (optional, based on confidence)
            if label == "Elephant" and conf > 0.8:
                send_alert()

    conn.close()
    return frame


def process_video(video_path, output_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Use the 'mp4v' codec for MP4 output (widely compatible)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Set up the video writer (make sure output is .mp4)
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    # Process frames from the input video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect objects in the current frame
        frame = detect_objects(frame)

        # Write the processed frame to the output video
        out.write(frame)

    # Release the video capture and writer objects
    cap.release()
    out.release()

def detect_video_frame(video_path):
    cap = cv2.VideoCapture(video_path)

    # Ensure video was opened correctly
    if not cap.isOpened():
        return "Error opening video file."

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply object detection (bounding boxes)
        frame = detect_objects(frame)

        # Encode frame as JPEG to stream it to the browser
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Return the frame as a response
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()



def gen_frames():
    global camera
    while is_camera_on:
        success, frame = camera.read()
        if not success:
            break
        frame = detect_objects(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def get_logs():
    conn = sqlite3.connect("detections.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM logs ORDER BY timestamp DESC LIMIT 10")
    logs = cursor.fetchall()
    conn.close()
    return logs

def get_alerts():
    conn = sqlite3.connect("detections.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM alerts ORDER BY timestamp DESC LIMIT 10")
    alerts = cursor.fetchall()
    conn.close()
    return alerts

def process_uploaded_video(video_path):
    cap = cv2.VideoCapture(video_path)

    # Ensure video was opened correctly
    if not cap.isOpened():
        return "Error opening video file."

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply object detection to the current frame
        frame = detect_objects(frame)

        # Encode the frame as JPEG to stream it to the browser
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame as a response
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()



@app.route('/')
def index():
    conn = sqlite3.connect("detections.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM logs ORDER BY timestamp DESC LIMIT 10")
    logs = cursor.fetchall()
    cursor.execute("SELECT * FROM alerts ORDER BY timestamp DESC LIMIT 10")
    alerts = cursor.fetchall()
    conn.close()
    return render_template('index4.html', logs=logs, alerts=alerts)

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get("file")
    if file:
        original_filename = "uploaded.jpg"
        detected_filename = "detected.jpg"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], original_filename)
        detected_path = os.path.join(app.config["UPLOAD_FOLDER"], detected_filename)

        file.save(file_path)
        frame = cv2.imread(file_path)
        result = detect_objects(frame)
        cv2.imwrite(detected_path, result)

        return render_template(
            "index4.html",
            uploaded_image=original_filename,
            detected_image=detected_filename,
            logs=get_logs(),
            alerts=get_alerts()
        )
    return redirect(url_for("index"))

# @app.route('/upload_video', methods=['POST'])
# def upload_video():
#     file = request.files['file']
    
#     if file and allowed_video(file.filename):
#         # Define paths
#         uploaded_video_folder = os.path.join(app.config["UPLOAD_FOLDER"], "uploaded_video")
#         detected_video_folder = os.path.join(app.config["UPLOAD_FOLDER"], "detected_video")

#         # Create directories if they don't exist
#         os.makedirs(uploaded_video_folder, exist_ok=True)
#         os.makedirs(detected_video_folder, exist_ok=True)

#         # Paths to save the video
#         original_path = os.path.join(uploaded_video_folder, "uploaded_video.mp4")
#         detected_path = os.path.join(detected_video_folder, "detected_video.mp4")

#         # Save the uploaded file
#         file.save(original_path)
        
#         # Process the video
#         process_video(original_path, detected_path)
        
#         return redirect(url_for('show_video_result'))
    
#     return "Invalid video format"

@app.route('/upload_video', methods=['POST'])
def upload_video():
    file = request.files['file']
    
    if file and allowed_video(file.filename):
        # Define path to save the uploaded video
        uploaded_video_folder = os.path.join(app.config["UPLOAD_FOLDER"], "uploaded_video")
        os.makedirs(uploaded_video_folder, exist_ok=True)
        original_path = os.path.join(uploaded_video_folder, "uploaded_video.mp4")
        
        # Save the uploaded video file
        file.save(original_path)
        
        # Now, stream video frames with detection
        return redirect(url_for('stream_uploaded_video', video_path=original_path))
    
    return "Invalid video format"

@app.route('/stream_uploaded_video')
def stream_uploaded_video():
    video_path = request.args.get('video_path')
    
    if not video_path:
        return "No video path provided."
    
    return Response(process_uploaded_video(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_result')
def show_video_result():
    return render_template("video_result.html")

@app.route('/start_cam', methods=['POST'])
def start_cam():
    global is_camera_on, camera
    if not is_camera_on:
        camera = cv2.VideoCapture(0)
        is_camera_on = True
    return jsonify({"status": "started"})

@app.route('/stop_cam', methods=['POST'])
def stop_cam():
    global is_camera_on, camera
    is_camera_on = False
    if camera:
        camera.release()
        camera = None
    return jsonify({"status": "stopped"})

@app.route('/video_feed')
def video_feed():
    if is_camera_on:
        return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Camera is off"


@app.route('/live')
def live():
    return render_template('live.html')

@app.route('/clear_logs', methods=['POST'])
def clear_logs():
    conn = sqlite3.connect("detections.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM logs")
    cursor.execute("DELETE FROM alerts")
    cursor.execute("DELETE FROM sqlite_sequence WHERE name='logs'")
    cursor.execute("DELETE FROM sqlite_sequence WHERE name='alerts'")
    conn.commit()
    conn.close()
    return redirect(url_for("index"))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)