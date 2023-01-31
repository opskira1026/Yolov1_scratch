import torch
# not change
def intersection_over_union(boxes_preds, boxes_labels, box_format='midpoint'):
    
    if box_format == 'midpoint':                                          # (middle_x, middle_y, width, height)
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
        
    if box_format == 'corners':                                           # (xmin, ymin, xmax, ymax)
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4] 
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]
    
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)                # 만일 0보다 작은 값이 나오면 clamp(0)을 사용하여 0으로 만들어줌.
    
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))            # 절댓값
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    
    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):

    assert type(bboxes) == list , "The type of bboxes must be a list."

    bboxes = [box for box in bboxes if box[1] > threshold]
    # 오름차순 정렬인데, lambda key인 x[1]이 probability score를 의미한다. 
    # sorted()를 사용하면 기본적으로 오름차순 정렬인데, 이러면 우리가 필요한 정보는 마지막에 위치하게 된다. 
    # 이를 reverse하게 되면 가장 높은 값이 맨 앞으로 오게 되므로 reverse = True를 사용한다.
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True) 
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)  # pop()를 사용한다는 것은 stack에 있는 값을 뺀다는 것을 의미한다.
                                    # 또한, pop(0)를 사용하면 list의 맨 앞의 값을 제거한다.(뺴낸다.)
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]
        
        # box[0]와 chosen_box[0]가 다르거나 intersection of union의 값이 iou_threshold 보다 작으면 bboxes에 넣음  
        # 즉, box[0]과 chosen_box[0]의 값이 같을 때까지 무한 반복함.

        bboxes_after_nms.append(chosen_box) # 이를 append 하여 값을 넣고 이를 반환시킴.

    return bboxes_after_nms

