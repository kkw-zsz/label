import redis
import megfile
import json
import numpy as np
import tqdm
import pickle

rd = redis.StrictRedis('localhost')
robot_name = 'aloha_5'
data_dir_base = 's3://dexmal-sharefs-pdd/300k_official'

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

if __name__ == '__main__':
    tasks = json.loads(rd.get('label_task_list_' + robot_name).decode('utf8'))
    dataset = []
    for i in tqdm.tqdm(range(100)):
        path = tasks[i]['path']
        frameid = tasks[i]['frameid']
        left_json = json.loads(megfile.smart_open(data_dir_base + '/' + path + '/data/' + robot_name + '_arms_left.jsonl', 'rb').read().decode('utf8').strip().splitlines()[frameid])
        right_json = json.loads(megfile.smart_open(data_dir_base + '/' + path + '/data/' + robot_name + '_arms_right.jsonl', 'rb').read().decode('utf8').strip().splitlines()[frameid])
        ee1 = left_json['ee_pose_quaternion']
        ee2 = right_json['ee_pose_quaternion']
        width1 = left_json['gripper']
        xyz1 = ee1[:3]
        rotmat1 = quaternion_to_rotmat(ee1[3:7])
        width2 = right_json['gripper']
        xyz2 = ee2[:3]
        rotmat2 = quaternion_to_rotmat(ee2[3:7])

        delta = 0.005
        est_point11 = xyz1 + rotmat1.dot(np.float32([0, width1/2 + delta, 0]))
        est_point12 = xyz1 + rotmat1.dot(np.float32([0, -width1/2 - delta, 0]))
        est_point1f = rotmat1.dot(np.float32([0, 0, 1]))
        feature1 = np.hstack([est_point11, est_point12, est_point1f])

        est_point21 = xyz2 + rotmat2.dot(np.float32([0, width2/2 + delta, 0]))
        est_point22 = xyz2 + rotmat2.dot(np.float32([0, -width2/2 - delta, 0]))
        est_point2f = rotmat2.dot(np.float32([0, 0, 1]))
        feature2 = np.hstack([est_point21, est_point22, est_point2f])

        labels = json.loads(rd.hget('label_result_' + path + '/videos/cam_high_rgb.mp4', frameid).decode('utf8'))
        dataset.append({
            'feature1' : feature1,
            'feature2' : feature2,
            **labels
        })
    fout = megfile.smart_open('s3://unsullied/sharefs/your_name/work/act/label/project2d/%s/data.pkl' % robot_name, 'wb')
    fout.write(pickle.dumps(dataset))
    fout.close()