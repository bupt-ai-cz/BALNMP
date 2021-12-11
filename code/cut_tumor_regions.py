# -*- coding: utf-8 -*
import numpy as np
import cv2
import os
import json
import glob
import argparse

FILL_COLOR = (255, 255, 255)  # [Red, Green, Blue], the solid colors to fill irrelevant areas


def get_annotation_points_and_bboxes(json_path):
    """get coordinate of each point in each annotated region, and get bounding box ([start_x, start_y, width, height]) of each annotated region"""
    with open(json_path) as f:
        asap_json = json.load(f)

    annotation_points = []
    bboxes = []

    # data only exist in 'positive'
    for i in asap_json['positive']:
        annotation_points.append(i['vertices'])
        bboxes.append(get_bbox(i['vertices']))

    return annotation_points, bboxes


def get_bbox(points):
    """get bounding box of an annotated region"""
    points = np.asarray(points)
    max_x_y = np.max(points, axis=0)
    min_x_y = np.min(points, axis=0)

    width_height = max_x_y - min_x_y
    bbox = min_x_y.tolist() + width_height.tolist()

    return bbox  # [start_x, start_y, width, height]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cut tumour regions of all WSIs')
    parser.add_argument('--wsi_dir_path', help='path of directory storing WSI and json', type=str, required=True)
    parser.add_argument('--output_dir_path', help='path of directory storing cut tumour regions', type=str, required=True)
    parser.add_argument('--not_filled_other_regions', help='not fill irrelevant areas with solid colors', action='store_false')
    args = parser.parse_args()

    assert not os.path.exists(args.output_dir_path), 'output_dir_path has existed, please change output_dir_path or remove it manually'

    # create output_dir_path
    os.makedirs(args.output_dir_path)
    print('create directory: {}'.format(args.output_dir_path))

    for wsi_path in glob.glob(os.path.join(args.wsi_dir_path, '*.jpg')):
        wsi_id = os.path.splitext(os.path.basename(wsi_path))[0]
        json_path = wsi_path.replace('jpg', 'json')

        # create directory to store tumour regions for each WSI
        os.makedirs(os.path.join(args.output_dir_path, wsi_id))
        print('create directory: {}'.format(os.path.join(args.output_dir_path, wsi_id)))

        annotation_points, bboxes = get_annotation_points_and_bboxes(json_path)

        wsi_img = cv2.imread(wsi_path)
        for i, (ann_points, bbox) in enumerate(zip(annotation_points, bboxes)):
            tumour_save_path = os.path.join(args.output_dir_path, wsi_id, '{}_{}.jpg'.format(wsi_id, i))

            # extract a rectangular tumour region
            x, y, width, height = bbox
            tumour_img = wsi_img[y: y + height, x: x + width]

            # fill irrelevant areas with solid colors
            if args.not_filled_other_regions:
                mask_array = np.zeros((height, width), dtype=np.uint8)

                ann_points = ann_points - np.asarray([x, y])  # compute the relative coordinate of each point in an annotated region
                ann_points = np.expand_dims(ann_points, 0)
                cv2.fillPoly(mask_array, ann_points, color=255)
                
                tumour_img[mask_array != 255] = FILL_COLOR  # fill irrelevant areas with solid colors

            cv2.imwrite(tumour_save_path, tumour_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            print('\t save {}'.format(tumour_save_path))
