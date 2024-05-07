import argparse
import asyncio
import json
import logging
import os
import platform
import ssl
import cv2
import base64
import numpy as np
import boto3
import pymysql
import mediapipe as mp

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer, MediaRelay
from aiortc.rtcrtpsender import RTCRtpSender

ROOT = os.path.dirname(__file__)


relay = None
webcam = None

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


def create_local_tracks(play_from, decode):
    global relay, webcam

    if play_from:
        player = MediaPlayer(play_from, decode=decode)
        return player.audio, player.video
    else:
        options = {"framerate": "30", "video_size": "640x480"}
        if relay is None:
            if platform.system() == "Darwin":
                webcam = MediaPlayer(
                    "default:none", format="avfoundation", options=options
                )
            elif platform.system() == "Windows":
                webcam = MediaPlayer(
                    "video=Integrated Camera", format="dshow", options=options
                )
            else:
                webcam = MediaPlayer("/dev/video0", format="v4l2", options=options)
            relay = MediaRelay()
        return None, relay.subscribe(webcam.video)


def force_codec(pc, sender, forced_codec):
    kind = forced_codec.split("/")[0]
    codecs = RTCRtpSender.getCapabilities(kind).codecs
    transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
    transceiver.setCodecPreferences(
        [codec for codec in codecs if codec.mimeType == forced_codec]
    )


async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)

async def analyze(request):
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

    data = await request.json()
    frame_data = data['frame']
    # Decode base64 string to image
    decoded_data = base64.b64decode(frame_data.split(',')[1])
    np_data = np.frombuffer(decoded_data, np.uint8)
    img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    img.flags.writeable = False
    try:
        results = face_mesh.process(img)
    except Exception as e:
        logging.error(f"Error processing frame: {e}")
            
    img.flags.writeable = True
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    (img_h, img_w, img_c) = img.shape
    face_2d = []

            # if not results.multi_face_landmarks:
            #     continue 

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
            print(data)
        else:
            store_gaze_data(connection,data)
            data=[]

            # cv2.imshow('Head Pose Estimation', img)

            # if cv2.waitKey(5) & 0xFF == ord('q'):
            #     break
    ret, buffer = cv2.imencode('.jpg', img)
    frame = buffer.tobytes()
    decoded_frame = cv2.imdecode(np.frombuffer(frame, np.uint8), -1)
    return web.Response(decoded_frame)

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    # open media source
    audio, video = create_local_tracks(
        args.play_from, decode=not args.play_without_decoding
    )

    if audio:
        audio_sender = pc.addTrack(audio)
        if args.audio_codec:
            force_codec(pc, audio_sender, args.audio_codec)
        elif args.play_without_decoding:
            raise Exception("You must specify the audio codec using --audio-codec")

    if video:
        video_sender = pc.addTrack(video)
        if args.video_codec:
            force_codec(pc, video_sender, args.video_codec)
        elif args.play_without_decoding:
            raise Exception("You must specify the video codec using --video-codec")

    await pc.setRemoteDescription(offer)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


pcs = set()


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC webcam demo")
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument("--play-from", help="Read the media from a file and sent it.")
    parser.add_argument(
        "--play-without-decoding",
        help=(
            "Read the media without decoding it (experimental). "
            "For now it only works with an MPEGTS container with only H.264 video."
        ),
        action="store_true",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    parser.add_argument(
        "--audio-codec", help="Force a specific audio codec (e.g. audio/opus)"
    )
    parser.add_argument(
        "--video-codec", help="Force a specific video codec (e.g. video/H264)"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/analyze", analyze)
    app.router.add_post("/offer", offer)
    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)