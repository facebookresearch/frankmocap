
import cv2
import math
from operator import itemgetter
import numpy as np
try:
    import onnxruntime
except ImportError:
    onnxruntime = None


class ORTWrapper:

    def __init__(self, onnx_f) -> None:
        self.onnx_f = onnx_f
        so = onnxruntime.SessionOptions()
        so.intra_op_num_threads = 6
        so.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.ort_session = onnxruntime.InferenceSession(
            onnx_f, sess_options=so)

        # todo: extract possible input hw for vision models

    def infer(self, imgs):
        inputs = [imgs]
        assert len(inputs) == len(self.ort_session.get_inputs()
                                  ), 'inputs must same with model.'
        ort_inputs = dict(
            (self.ort_session.get_inputs()[i].name, inpt) for i, inpt in enumerate(inputs))
        ort_outs = self.ort_session.run(None, ort_inputs)

        outs_dict = dict()
        for i, oo in enumerate(self.ort_session.get_outputs()):
            n = oo.name
            outs_dict[n] = ort_outs[i]
        return outs_dict


BODY_PARTS_KPT_IDS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
                      [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 16], [5, 17]]
BODY_PARTS_PAF_IDS = ([12, 13], [20, 21], [14, 15], [16, 17], [22, 23], [24, 25], [0, 1], [2, 3], [4, 5],
                      [6, 7], [8, 9], [10, 11], [28, 29], [30, 31], [34, 35], [32, 33], [36, 37], [18, 19], [26, 27])


def normalize(img, img_mean, img_scale):
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img


def pad_width(img, stride, pad_value, min_dims):
    h, w, _ = img.shape
    pad = []
    pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
    pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
                                    cv2.BORDER_CONSTANT, value=pad_value)
    return padded_img, pad


def connections_nms(a_idx, b_idx, affinity_scores):
    # From all retrieved connections that share the same starting/ending keypoints leave only the top-scoring ones.
    order = affinity_scores.argsort()[::-1]
    affinity_scores = affinity_scores[order]
    a_idx = a_idx[order]
    b_idx = b_idx[order]
    idx = []
    has_kpt_a = set()
    has_kpt_b = set()
    for t, (i, j) in enumerate(zip(a_idx, b_idx)):
        if i not in has_kpt_a and j not in has_kpt_b:
            idx.append(t)
            has_kpt_a.add(i)
            has_kpt_b.add(j)
    idx = np.asarray(idx, dtype=np.int32)
    return a_idx[idx], b_idx[idx], affinity_scores[idx]


def group_keypoints(all_keypoints_by_type, pafs, pose_entry_size=20, min_paf_score=0.05):
    pose_entries = []
    all_keypoints = np.array(
        [item for sublist in all_keypoints_by_type for item in sublist])
    points_per_limb = 10
    grid = np.arange(points_per_limb, dtype=np.float32).reshape(1, -1, 1)
    all_keypoints_by_type = [np.array(keypoints, np.float32)
                             for keypoints in all_keypoints_by_type]

    for part_id in range(len(BODY_PARTS_PAF_IDS)):
        part_pafs = pafs[:, :, BODY_PARTS_PAF_IDS[part_id]]
        kpts_a = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][0]]
        kpts_b = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][1]]
        n = len(kpts_a)
        m = len(kpts_b)
        if n == 0 or m == 0:
            continue

        # Get vectors between all pairs of keypoints, i.e. candidate limb vectors.
        a = kpts_a[:, :2]
        a = np.broadcast_to(a[None], (m, n, 2))
        b = kpts_b[:, :2]
        vec_raw = (b[:, None, :] - a).reshape(-1, 1, 2)

        # Sample points along every candidate limb vector.
        steps = (1 / (points_per_limb - 1) * vec_raw)
        points = steps * grid + a.reshape(-1, 1, 2)
        points = points.round().astype(dtype=np.int32)
        x = points[..., 0].ravel()
        y = points[..., 1].ravel()

        # Compute affinity score between candidate limb vectors and part affinity field.
        field = part_pafs[y, x].reshape(-1, points_per_limb, 2)
        vec_norm = np.linalg.norm(vec_raw, ord=2, axis=-1, keepdims=True)
        vec = vec_raw / (vec_norm + 1e-6)
        affinity_scores = (field * vec).sum(-1).reshape(-1, points_per_limb)
        valid_affinity_scores = affinity_scores > min_paf_score
        valid_num = valid_affinity_scores.sum(1)
        affinity_scores = (affinity_scores *
                           valid_affinity_scores).sum(1) / (valid_num + 1e-6)
        success_ratio = valid_num / points_per_limb

        # Get a list of limbs according to the obtained affinity score.
        valid_limbs = np.where(np.logical_and(
            affinity_scores > 0, success_ratio > 0.8))[0]
        if len(valid_limbs) == 0:
            continue
        b_idx, a_idx = np.divmod(valid_limbs, n)
        affinity_scores = affinity_scores[valid_limbs]

        # Suppress incompatible connections.
        a_idx, b_idx, affinity_scores = connections_nms(
            a_idx, b_idx, affinity_scores)
        connections = list(zip(kpts_a[a_idx, 3].astype(np.int32),
                               kpts_b[b_idx, 3].astype(np.int32),
                               affinity_scores))
        if len(connections) == 0:
            continue

        if part_id == 0:
            pose_entries = [np.ones(pose_entry_size)
                            * -1 for _ in range(len(connections))]
            for i in range(len(connections)):
                pose_entries[i][BODY_PARTS_KPT_IDS[0][0]] = connections[i][0]
                pose_entries[i][BODY_PARTS_KPT_IDS[0][1]] = connections[i][1]
                pose_entries[i][-1] = 2
                pose_entries[i][-2] = np.sum(
                    all_keypoints[connections[i][0:2], 2]) + connections[i][2]
        elif part_id == 17 or part_id == 18:
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            for i in range(len(connections)):
                for j in range(len(pose_entries)):
                    if pose_entries[j][kpt_a_id] == connections[i][0] and pose_entries[j][kpt_b_id] == -1:
                        pose_entries[j][kpt_b_id] = connections[i][1]
                    elif pose_entries[j][kpt_b_id] == connections[i][1] and pose_entries[j][kpt_a_id] == -1:
                        pose_entries[j][kpt_a_id] = connections[i][0]
            continue
        else:
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            for i in range(len(connections)):
                num = 0
                for j in range(len(pose_entries)):
                    if pose_entries[j][kpt_a_id] == connections[i][0]:
                        pose_entries[j][kpt_b_id] = connections[i][1]
                        num += 1
                        pose_entries[j][-1] += 1
                        pose_entries[j][-2] += all_keypoints[connections[i]
                                                             [1], 2] + connections[i][2]
                if num == 0:
                    pose_entry = np.ones(pose_entry_size) * -1
                    pose_entry[kpt_a_id] = connections[i][0]
                    pose_entry[kpt_b_id] = connections[i][1]
                    pose_entry[-1] = 2
                    pose_entry[-2] = np.sum(all_keypoints[connections[i]
                                            [0:2], 2]) + connections[i][2]
                    pose_entries.append(pose_entry)

    filtered_entries = []
    for i in range(len(pose_entries)):
        if pose_entries[i][-1] < 3 or (pose_entries[i][-2] / pose_entries[i][-1] < 0.2):
            continue
        filtered_entries.append(pose_entries[i])
    pose_entries = np.asarray(filtered_entries)
    return pose_entries, all_keypoints


def extract_keypoints(heatmap, all_keypoints, total_keypoint_num):
    heatmap[heatmap < 0.1] = 0
    heatmap_with_borders = np.pad(heatmap, [(2, 2), (2, 2)], mode='constant')
    heatmap_center = heatmap_with_borders[1:heatmap_with_borders.shape[0] -
                                          1, 1:heatmap_with_borders.shape[1]-1]
    heatmap_left = heatmap_with_borders[1:heatmap_with_borders.shape[0] -
                                        1, 2:heatmap_with_borders.shape[1]]
    heatmap_right = heatmap_with_borders[1:heatmap_with_borders.shape[0] -
                                         1, 0:heatmap_with_borders.shape[1]-2]
    heatmap_up = heatmap_with_borders[2:heatmap_with_borders.shape[0],
                                      1:heatmap_with_borders.shape[1]-1]
    heatmap_down = heatmap_with_borders[0:heatmap_with_borders.shape[0] -
                                        2, 1:heatmap_with_borders.shape[1]-1]

    heatmap_peaks = (heatmap_center > heatmap_left) &\
                    (heatmap_center > heatmap_right) &\
                    (heatmap_center > heatmap_up) &\
                    (heatmap_center > heatmap_down)
    heatmap_peaks = heatmap_peaks[1:heatmap_center.shape[0] -
                                  1, 1:heatmap_center.shape[1]-1]
    keypoints = list(zip(np.nonzero(heatmap_peaks)[
                     1], np.nonzero(heatmap_peaks)[0]))  # (w, h)
    keypoints = sorted(keypoints, key=itemgetter(0))

    suppressed = np.zeros(len(keypoints), np.uint8)
    keypoints_with_score_and_id = []
    keypoint_num = 0
    for i in range(len(keypoints)):
        if suppressed[i]:
            continue
        for j in range(i+1, len(keypoints)):
            if math.sqrt((keypoints[i][0] - keypoints[j][0]) ** 2 +
                         (keypoints[i][1] - keypoints[j][1]) ** 2) < 6:
                suppressed[j] = 1
        keypoint_with_score_and_id = (keypoints[i][0], keypoints[i][1], heatmap[keypoints[i][1], keypoints[i][0]],
                                      total_keypoint_num + keypoint_num)
        keypoints_with_score_and_id.append(keypoint_with_score_and_id)
        keypoint_num += 1
    all_keypoints.append(keypoints_with_score_and_id)
    return keypoint_num
