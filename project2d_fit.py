import numpy as np
import megfile
import pickle
import matplotlib.pyplot as plt

robot_name = 'aloha_5'

if __name__ == '__main__':
    with megfile.smart_open('s3://unsullied/sharefs/your_name/work/act/label/project2d/%s/data.pkl'%robot_name, 'rb') as fin:
        dataset = pickle.loads(fin.read())
    regvec = np.zeros((13, 4))
    selected_id = [1, 2, 5, 6]
    for pt_idx_base in range(0, 4, 2):
        xs = []
        ys = []
        original_ids = []
        for pt_idx in range(pt_idx_base, pt_idx_base + 2):
            for taskid, data in enumerate(dataset):
                if 'point%d'%selected_id[pt_idx] in data:
                    ys.append(data['point%d'%selected_id[pt_idx]])
                    if pt_idx == 0:
                        xyzvec = np.hstack([data['feature1'][:3], data['feature1'][6:9]])
                    elif pt_idx == 1:
                        xyzvec = np.hstack([data['feature1'][3:6], data['feature1'][6:9]])
                    elif pt_idx == 2:
                        xyzvec = np.hstack([data['feature2'][:3], data['feature2'][6:9]])
                    else:
                        xyzvec = np.hstack([data['feature2'][3:6], data['feature2'][6:9]])
                    xs.append(xyzvec)
                    original_ids.append(taskid)
        xs = np.float32(xs)
        ys = np.float32(ys)

        preds = np.zeros((xs.shape[0], 2))
        for i in range(2):
            regvec[:, pt_idx_base + i] = np.linalg.lstsq(np.hstack([
                xs,
                -xs * ys[:, i:i + 1],
                np.ones((xs.shape[0], 1))
            ]), ys[:, i])[0]
            preds[:, i] = (xs.dot(regvec[:6, pt_idx_base + i]) + regvec[12, pt_idx_base + i]) / (1 + xs.dot(regvec[6:12, pt_idx_base + i]))

        color = 'rgbcmykr'[pt_idx]
        plt.plot([ys[j][0] for j in range(len(xs))], [ys[j][1] for j in range(len(xs))], color + '.')
        for j in range(len(xs)):
            plt.plot([ys[j][0], preds[j][0]], [ys[j][1], preds[j][1]], color + '-')
        loss = np.abs(np.float32(preds) - ys)
        print(pt_idx, len(xs), 'loss', loss.mean())
        print('worst', original_ids[loss.mean(axis=1).argmax()])
    fout = megfile.smart_open('s3://unsullied/sharefs/your_name/work/act/label/project2d/%s/fit.pkl'%robot_name, 'wb')
    pickle.dump(regvec, fout)
    fout.close()
    plt.xlim(0, 1)
    plt.ylim(1, 0)
    plt.gcf().set_size_inches(12, 8)
    plt.savefig('project2d.png')