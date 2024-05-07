from flask import Flask, render_template, Response, request, jsonify
import logging
import cv2
import numpy as np
import mediapipe as mp
import pymysql
import asyncio
import numpy as np
from io import BytesIO
import boto3

app = Flask(__name__, static_url_path='/static')
def db_connection():

    boto3.setup_default_session(
    aws_access_key_id='',
    aws_secret_access_key='',
    region_name='')
    

    connection = pymysql.connect(host='51.20.91.6',
                             user='your_user',
                             password='your_password',
                             db='procting_db',
                             port=3306)
    
    return connection
def store_gaze_data(connection, data):
    try:
        mycursor = connection.cursor()
        sql = "INSERT INTO gaze_data (left_gaze, right_gaze) VALUES (%s, %s)"
        val = data
        mycursor.executemany(sql, val)
        connection.commit()
        print(mycursor.rowcount, "record inserted.")

    except Exception as e:
        logging.error(f"Error storing gaze data: {e}")

# Function to generate frames for video streaming
def generate_frames():
    connection = db_connection()
    data=[]
    batch=[]
    draw_gaze = True
    draw_full_axis = True
    draw_headpose = False

    x_score_multiplier = 10
    y_score_multiplier = 10
    threshold = .8

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
        refine_landmarks=True,
        max_num_faces=2,
        min_detection_confidence=0.5)


    face_3d = np.array([
        [0.0, 0.0, 0.0],          
        [0.0, -330.0, -65.0],  
        [-225.0, 170.0, -135.0],   
        [225.0, 170.0, -135.0],    
        [-150.0, -150.0, -125.0],   
        [150.0, -150.0, -125.0]     
        ], dtype=np.float64)

    leye_3d = np.array(face_3d)
    leye_3d[:,0] += 225
    leye_3d[:,1] -= 175
    leye_3d[:,2] += 135

    reye_3d = np.array(face_3d)
    reye_3d[:,0] -= 225
    reye_3d[:,1] -= 175
    reye_3d[:,2] += 135

    last_lx, last_rx = 0, 0
    last_ly, last_ry = 0, 0

    try:
        cap = cv2.VideoCapture(0)
        frame_count = 0
        while cap.isOpened():

            success, img = cap.read()
            img.flags.writeable = False

            try:
                results = face_mesh.process(img)
            except Exception as e:
                logging.error(f"Error processing frame: {e}")
                continue
            
            img.flags.writeable = True
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            (img_h, img_w, img_c) = img.shape
            face_2d = []

            if not results.multi_face_landmarks:
                continue 

            for face_landmarks in results.multi_face_landmarks:
                face_2d = []
                for idx, lm in enumerate(face_landmarks.landmark):
                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    face_2d.append((x, y))

                face_2d_head = np.array([
                    face_2d[1],    
                    face_2d[199],   
                    face_2d[33],    
                    face_2d[263],   
                    face_2d[61],    
                    face_2d[291]   
                ], dtype=np.float64)

                face_2d = np.asarray(face_2d)

                if (face_2d[243,0] - face_2d[130,0]) != 0:
                    lx_score = (face_2d[468,0] - face_2d[130,0]) / (face_2d[243,0] - face_2d[130,0])
                    if abs(lx_score - last_lx) < threshold:
                        lx_score = (lx_score + last_lx) / 2
                    last_lx = lx_score

                if (face_2d[23,1] - face_2d[27,1]) != 0:
                    ly_score = (face_2d[468,1] - face_2d[27,1]) / (face_2d[23,1] - face_2d[27,1])
                    if abs(ly_score - last_ly) < threshold:
                        ly_score = (ly_score + last_ly) / 2
                    last_ly = ly_score

                if (face_2d[359,0] - face_2d[463,0]) != 0:
                    rx_score = (face_2d[473,0] - face_2d[463,0]) / (face_2d[359,0] - face_2d[463,0])
                    if abs(rx_score - last_rx) < threshold:
                        rx_score = (rx_score + last_rx) / 2
                    last_rx = rx_score

                if (face_2d[253,1] - face_2d[257,1]) != 0:
                    ry_score = (face_2d[473,1] - face_2d[257,1]) / (face_2d[253,1] - face_2d[257,1])
                    if abs(ry_score - last_ry) < threshold:
                        ry_score = (ry_score + last_ry) / 2
                    last_ry = ry_score

                focal_length = 1 * img_w
                cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                        [0, focal_length, img_w / 2],
                                        [0, 0, 1]])

                dist_coeffs = np.zeros((4, 1), dtype=np.float64)

                _, l_rvec, l_tvec = cv2.solvePnP(leye_3d, face_2d_head, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                _, r_rvec, r_tvec = cv2.solvePnP(reye_3d, face_2d_head, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

                l_rmat, _ = cv2.Rodrigues(l_rvec)
                r_rmat, _ = cv2.Rodrigues(r_rvec)

                l_gaze_rvec = np.array(l_rvec)
                l_gaze_rvec[2][0] -= (lx_score-.5) * x_score_multiplier
                l_gaze_rvec[0][0] += (ly_score-.5) * y_score_multiplier

                r_gaze_rvec = np.array(r_rvec)
                r_gaze_rvec[2][0] -= (rx_score-.5) * x_score_multiplier
                r_gaze_rvec[0][0] += (ry_score-.5) * y_score_multiplier


                l_corner = face_2d_head[2].astype(np.int32)

                axis = np.float32([[-100, 0, 0], [0, 100, 0], [0, 0, 300]]).reshape(-1, 3)
                l_axis, _ = cv2.projectPoints(axis, l_rvec, l_tvec, cam_matrix, dist_coeffs)
                l_gaze_axis, _ = cv2.projectPoints(axis, l_gaze_rvec, l_tvec, cam_matrix, dist_coeffs)


                if draw_headpose:
                    if draw_full_axis:
                        cv2.line(img, l_corner, tuple(np.ravel(l_axis[0]).astype(np.int32)), (200,200,0), 3)
                        cv2.line(img, l_corner, tuple(np.ravel(l_axis[1]).astype(np.int32)), (0,200,0), 3)
                    cv2.line(img, l_corner, tuple(np.ravel(l_axis[2]).astype(np.int32)), (0,200,200), 3)

                if draw_gaze:
                    if draw_full_axis:
                        cv2.line(img, l_corner, tuple(np.ravel(l_gaze_axis[0]).astype(np.int32)), (255,0,0), 3)
                        cv2.line(img, l_corner, tuple(np.ravel(l_gaze_axis[1]).astype(np.int32)), (0,255,0), 3)
                    cv2.line(img, l_corner, tuple(np.ravel(l_gaze_axis[2]).astype(np.int32)), (0,0,255), 3)

                

                r_corner = face_2d_head[3].astype(np.int32)


                r_axis, _ = cv2.projectPoints(axis, r_rvec, r_tvec, cam_matrix, dist_coeffs)
                r_gaze_axis, _ = cv2.projectPoints(axis, r_gaze_rvec, r_tvec, cam_matrix, dist_coeffs)

                if draw_headpose:
                    if draw_full_axis:
                        cv2.line(img, r_corner, tuple(np.ravel(r_axis[0]).astype(np.int32)), (200,200,0), 3)
                        cv2.line(img, r_corner, tuple(np.ravel(r_axis[1]).astype(np.int32)), (0,200,0), 3)
                    cv2.line(img, r_corner, tuple(np.ravel(r_axis[2]).astype(np.int32)), (0,200,200), 3)

                if draw_gaze:
                    if draw_full_axis:
                        cv2.line(img, r_corner, tuple(np.ravel(r_gaze_axis[0]).astype(np.int32)), (255,0,0), 3)
                        cv2.line(img, r_corner, tuple(np.ravel(r_gaze_axis[1]).astype(np.int32)), (0,255,0), 3)
                    cv2.line(img, r_corner, tuple(np.ravel(r_gaze_axis[2]).astype(np.int32)), (0,0,255), 3)
                        

                if l_gaze_rvec[2][0] < -threshold:
                    l_gaze_x = 'right'
                elif l_gaze_rvec[2][0] > threshold+0.5:
                    l_gaze_x = 'left'
                else:
                    l_gaze_x = 'center'

                if l_gaze_rvec[0][0] < -threshold:
                    l_gaze_y = 'up'
                elif l_gaze_rvec[0][0] > threshold:
                    l_gaze_y = 'down'
                else:
                    l_gaze_y = 'center'
                if r_gaze_rvec[2][0] < -threshold:
                    r_gaze_x = 'right'
                elif r_gaze_rvec[2][0] > threshold:
                    r_gaze_x = 'left'
                else:
                    r_gaze_x = 'center'

                if r_gaze_rvec[0][0] < -threshold:
                    r_gaze_y = 'up'
                elif r_gaze_rvec[0][0] > threshold:
                    r_gaze_y = 'down'
                else:
                    r_gaze_y = 'center'

                    
                # cv2.putText(img, l_gaze_x, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # cv2.putText(img, r_gaze_x, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if len(data) < 100:
                    data.append((l_gaze_x,r_gaze_x))
                else:
                    store_gaze_data(connection,data)
                    data=[]

            # cv2.imshow('Head Pose Estimation', img)

            # if cv2.waitKey(5) & 0xFF == ord('q'):
            #     break
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            decoded_frame = cv2.imdecode(np.frombuffer(frame, np.uint8), -1)
            yield decoded_frame

    except Exception as e:
        logging.error(f"Error generating frames: {e}")
        cap.release()
        # cv2.destroyAllWindows()

# Route to stream video frames
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return "The health is good"
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')