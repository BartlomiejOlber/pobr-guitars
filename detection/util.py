import numpy as np


def select_best(boxes: np.ndarray, scores: np.ndarray, threshold: float):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    box_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    remaining_ids = np.argsort(scores)

    while len(remaining_ids) > 0:
        best_remaining = len(remaining_ids) - 1
        best_remaining_id = remaining_ids[best_remaining]
        pick.append(best_remaining_id)

        x1_max = np.maximum(x1[best_remaining_id], x1[remaining_ids[:best_remaining]])
        y1_max = np.maximum(y1[best_remaining_id], y1[remaining_ids[:best_remaining]])
        x2_min = np.minimum(x2[best_remaining_id], x2[remaining_ids[:best_remaining]])
        y2_min = np.minimum(y2[best_remaining_id], y2[remaining_ids[:best_remaining]])

        overlap_width = np.maximum(0, x2_min - x1_max + 1)
        overlap_height = np.maximum(0, y2_min - y1_max + 1)

        overlap = (overlap_width * overlap_height) / box_area[remaining_ids[:best_remaining]]

        remaining_ids = np.delete(remaining_ids, np.concatenate(([best_remaining],
                                                                 np.where(overlap > threshold)[0])))

    return boxes[pick].astype("int")
