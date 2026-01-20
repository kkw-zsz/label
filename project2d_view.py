import av
import cv2
import numpy as np
import pickle
import json
import megfile
import tqdm

robot_name = 'aloha_5'
data_dir_base = 's3://dexmal-sharefs-pdd/300k_official'

episode = 's3://dexmal-sharefs-pdd/300k_official/aloha_5/2025_08_18/pour_the_french_fries_into_the_plate/180610/v1/'

regvec = pickle.load(megfile.smart_open('s3://unsullied/sharefs/your_name/work/act/label/project2d/%s/fit.pkl'%robot_name, 'rb'))
#  (13, 4)

def quaternion_to_rotmat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]])
#   vec_cur = rotmat.dot(vec_initial)
#         z   x
#         |  /
#         | /
#         |/
#   y---- +

def extract_endpoints(ee, gripper, reg):
    xyz1 = ee[:3]
    rotmat1 = quaternion_to_rotmat(ee[3:7])
    width1 = gripper
    delta = 0.005
    est_point11 = xyz1 + rotmat1.dot(np.float32([0, width1/2 + delta, 0]))
    est_point12 = xyz1 + rotmat1.dot(np.float32([0, -width1/2 - delta, 0]))
    est_point1f = rotmat1.dot(np.float32([0, 0, 1]))

    preds = np.float32([
        [
            (np.hstack([est_point11, est_point1f]).dot(reg[:6, 0]) + reg[12, 0]) / (1 + np.hstack([est_point11, est_point1f]).dot(reg[6:12, 0])),
            (np.hstack([est_point11, est_point1f]).dot(reg[:6, 1]) + reg[12, 1]) / (1 + np.hstack([est_point11, est_point1f]).dot(reg[6:12, 1])),
        ],
        [
            (np.hstack([est_point12, est_point1f]).dot(reg[:6, 0]) + reg[12, 0]) / (1 + np.hstack([est_point12, est_point1f]).dot(reg[6:12, 0])),
            (np.hstack([est_point12, est_point1f]).dot(reg[:6, 1]) + reg[12, 1]) / (1 + np.hstack([est_point12, est_point1f]).dot(reg[6:12, 1])),
        ],
    ])
    return preds

if __name__ == '__main__':
    video_name = episode + 'videos/cam_high_rgb.mp4'
    container = av.open(megfile.smart_open(video_name, 'rb'))
    json1 = [json.loads(i) for i in megfile.smart_open(episode + 'data/%s_arms_left.jsonl' % robot_name, 'r').read().strip().splitlines()]
    json2 = [json.loads(i) for i in megfile.smart_open(episode + 'data/%s_arms_right.jsonl' % robot_name, 'r').read().strip().splitlines()]

    output_container = av.open('project2d_view.mp4', 'w')
    stream = output_container.add_stream('h264', rate=30)
    stream.width = 640
    stream.height = 480
    stream.pix_fmt = 'yuv420p'

    history_points = []

    for frameid, frame in tqdm.tqdm(enumerate(container.decode(video=0))):
        img = frame.to_rgb().to_ndarray()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (640, 480))

        ee1 = json1[frameid]['ee_pose_quaternion']
        gripper1 = json1[frameid]['gripper']
        endpoints1 = extract_endpoints(ee1, gripper1, regvec[:, :2])
        ee2 = json2[frameid]['ee_pose_quaternion']
        gripper2 = json2[frameid]['gripper']
        endpoints2 = extract_endpoints(ee2, gripper2, regvec[:, 2:4])

        endpoints = np.vstack([endpoints1, endpoints2])

        colors = [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
        ]
        for h in history_points:
            for p_id, (x, y) in enumerate(h):
                color = colors[p_id]
                cv2.circle(img,
                    (int(img.shape[1] * x), int(img.shape[0] * y)), 0, color, 1
                )

        for p_id, (x, y) in enumerate(endpoints):
            color = colors[p_id]
            cv2.circle(img, 
                (int(img.shape[1] * x), int(img.shape[0] * y)), 3, color , 1
            )
        
        history_points.append(endpoints)
        if len(history_points) >= 15:
            history_points = history_points[-15:]

        # Convert BGR to RGB for PyAV
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_out = av.VideoFrame.from_ndarray(img_rgb, format='rgb24')
        for packet in stream.encode(frame_out):
            output_container.mux(packet)
    # Flush stream
    for packet in stream.encode():
        output_container.mux(packet)
    output_container.close()