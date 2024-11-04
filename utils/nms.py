

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils.plot_utils import plot_bboxes
from datasets.yolo_dataset import reconstruct_raw_labels_threshold
from iou import ioU_NMS

def low_confidence_suppression(label):
    label = torch.where(label[0,:,:] < 0.6, label,0)
    return label

def non_max_suppression(label,nms_thres=0.5):
    label_w_conf = reconstruct_raw_labels_threshold(label.cpu().detach().numpy(), include_score=True, threshold=0.6)
    # label_w_conf = reconstruct_raw_labels(label, conf=True, thr=0.6)


    nms_batch = []
    for i, each_label in enumerate(label_w_conf):       # for each image in the batch
        keep_lst_car = []
        keep_lst_tra = []
        keep_lst_ped = []
        ind = np.argsort(each_label[:,-1])
        sorted_each_label = each_label[ind]

        car = [k for k in sorted_each_label if k[0]==2.][::-1]
        traffic_light = [k for k in sorted_each_label if k[0]==1.][::-1]
        person = [k for k in sorted_each_label if k[0]==0.][::-1]

        # print("Car : ", car)
        # print("TL : ", traffic_light)
        # print("Person : ", person)
        car_max = []
        traffic_light_max = []
        person_max = []

        while car != []:
            car_max.append(car.pop(0))
            first = car_max[-1]
            for other_cars in car:
                if ioU_NMS(other_cars, first) > nms_thres:
                    array_to_remove = other_cars
                    car = [a for a, skip in zip(car, [np.allclose(a, array_to_remove) for a in car]) if not skip]


        while traffic_light != []:
            traffic_light_max.append(traffic_light.pop(0))
            first = traffic_light_max[-1]
            for other_traffic_light in traffic_light:
                if ioU_NMS(other_traffic_light, first) > nms_thres:
                    array_to_remove = other_traffic_light
                    traffic_light = [a for a, skip in zip(traffic_light, [np.allclose(a, array_to_remove) for a in traffic_light]) if not skip]

        while person != []:
            person_max.append(person.pop(0))
            first = person_max[-1]
            for other_person in person:
                if ioU_NMS(other_person, first) > nms_thres:
                    array_to_remove = other_person
                    person = [a for a, skip in zip(person, [np.allclose(a, array_to_remove) for a in person]) if not skip]

        nms_batch.append([person_max, traffic_light_max, car_max])
    # print("NMS : ", nms_batch)

    return nms_batch

def non_max_suppression_threshold(label,nms_thres = 0.5):
    label_w_conf = reconstruct_raw_labels_threshold(label.cpu().detach().numpy(), include_score=True, threshold=0.6)
    # label_w_conf = reconstruct_raw_labels(label, conf=True, thr=0.6)


    nms_batch = []
    for i, each_label in enumerate(label_w_conf):       # for each image in the batch
        keep_lst_car = []
        keep_lst_tra = []
        keep_lst_ped = []
        ind = np.argsort(each_label[:,-1])
        sorted_each_label = each_label[ind]

        car = [k for k in sorted_each_label if k[0]==2.][::-1]
        traffic_light = [k for k in sorted_each_label if k[0]==1.][::-1]
        person = [k for k in sorted_each_label if k[0]==0.][::-1]

        # print("Car : ", car)
        # print("TL : ", traffic_light)
        # print("Person : ", person)
        car_max = []
        traffic_light_max = []
        person_max = []

        while car != []:
            car_max.append(car.pop(0))
            first = car_max[-1]
            for other_cars in car:
                if ioU_NMS(other_cars, first) > nms_thres:
                    array_to_remove = other_cars
                    car = [a for a, skip in zip(car, [np.allclose(a, array_to_remove) for a in car]) if not skip]


        while traffic_light != []:
            traffic_light_max.append(traffic_light.pop(0))
            first = traffic_light_max[-1]
            for other_traffic_light in traffic_light:
                if ioU_NMS(other_traffic_light, first) > nms_thres:
                    array_to_remove = other_traffic_light
                    traffic_light = [a for a, skip in zip(traffic_light, [np.allclose(a, array_to_remove) for a in traffic_light]) if not skip]

        while person != []:
            person_max.append(person.pop(0))
            first = person_max[-1]
            for other_person in person:
                if ioU_NMS(other_person, first) > nms_thres:
                    array_to_remove = other_person
                    person = [a for a, skip in zip(person, [np.allclose(a, array_to_remove) for a in person]) if not skip]

        nms_batch.append([person_max, traffic_light_max, car_max])
    # print("NMS : ", nms_batch)

    return nms_batch