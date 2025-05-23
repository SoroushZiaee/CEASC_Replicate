# a script for organizing the dataset to COCO format, the same format that VisDrone is in and that the model was designed 
# for.
# This script was inherited from https://github.com/PuAnysh/UFPMP-Det, the repository that Du et al., 2023 recommended 
# downloading the dataset from. The link to the pre-processed dataset was dead and the authors of the dataset (Du et al., 2018)
# provided no indication as to how to convert their dataset to the necessary format. 

import argparse
import os
import cv2
import json


def parse_args():
    parser = argparse.ArgumentParser(description='build UFP images')
    parser.add_argument('dataset_root', help='the dir to save logs and models') # this had to be changed to get the code to work
    parser.add_argument('output_anno', help='the dir to save logs and models')
    parser.add_argument('exclude_set', help='which set to not organize')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    global_image = 0
    global_anno = 0

    root = args.dataset_root
    img_prefix = 'img{:0>6d}.jpg'

    testset = ['M0203', 'M0205', 'M0208', 'M0209', 'M0403', 'M0601', 'M0602', 'M0606',
            'M0701', 'M0801', 'M0802', 'M1001',
            'M1004', 'M1007', 'M1009', 'M1101', 'M1301', 'M1302', 'M1303', 'M1401']
    
    trainset = ['M0902', 'M1006', 'M1202', 'M1305', 'M0202', 'M0101', 'M0301', 'M0206', 
                'M0703', 'M0604', 'M0901', 'M1008', 'M1002', 'M1306', 'M1201', 'M1102', 
                'M1005', 'M0704', 'M0603', 'M0201', 'M1304', 'M0210', 'M0204', 'M0402', 
                'M0501', 'M1003', 'M0207', 'M0605', 'M0702', 'M0401']
    
    set_dict = {'testset':testset, 'trainset': trainset}

    json_label = {
        'images': [],
        'type': 'instances',
        'annotations': [],
        'categories': [{'supercategory': 'none', 'id': 0, 'name': 'car'},
                    {'supercategory': 'none', 'id': 1, 'name': 'truck'},
                    {'supercategory': 'none', 'id': 2, 'name': 'bus'}, ]
    }


    def build_image(file_name, height, width, id):
        image = dict()
        image['file_name'] = file_name
        image['height'] = height
        image['width'] = width
        image['id'] = id
        return image


    def build_annotation(xmin, ymin, o_width, o_height, image_id, bnd_id, category_id, ignore=0):
        xmax = xmin + o_width
        ymax = ymin + o_height
        annotation = dict()
        annotation['area'] = o_width * o_height
        annotation['iscrowd'] = 0
        annotation['image_id'] = image_id
        annotation['bbox'] = [xmin, ymin, o_width, o_height]
        annotation['category_id'] = category_id
        annotation['id'] = bnd_id
        annotation['ignore'] = ignore
        # 设置分割数据，点的顺序为逆时针方向
        annotation['segmentation'] = [[xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]]
        return annotation


    def get_gt_by_frame(gt_path):
        gts = {}
        gt = open(gt_path)
        for items in gt.readlines():
            items = [int(_) for _ in items.split(',')]
            [frame_index, target_id, bbox_left, bbox_top, bbox_width, bbox_height, out_of_view, occlusion,
            object_category] = items
            if frame_index not in gts.keys():
                gts[frame_index] = []
            gts[frame_index].append([bbox_left, bbox_top, bbox_width, bbox_height, object_category - 1])
        return gts
    
    root_list = os.listdir(root)
    root_list.sort()

    for _dir in root_list:
        if _dir in set_dict[args.exclude_set]: # our modification to ensure that we can use the same script to organize both the train and test subsets 
            continue
        if _dir == "GT": # our modification so that the code does not break when it hits this directory 
            continue

        # print(f"processing {_dir}")

        gt_path = os.path.join(root,'GT',f'{_dir}_gt_whole.txt') # our modification to make this work with our storage style for this dataset 
        gt_ignore_path = os.path.join(root,'GT',f'{_dir}_gt_ignore.txt')

        # gt_path = os.path.join(root, _dir, 'gt', 'gt_whole.txt')
        # gt_ignore_path = os.path.join(root, _dir, 'gt', 'gt_ignore.txt')

        gts = get_gt_by_frame(gt_path)
        gts_ignore = get_gt_by_frame(gt_ignore_path)
        for frame_id in gts.keys():
            # img_path = os.path.join(root, _dir, 'img1', img_prefix.format(frame_id))
            img_path = os.path.join(root, _dir, img_prefix.format(frame_id))
            image_data = cv2.imread(img_path)
            h, w, c = image_data.shape
            # image_meta = build_image(os.path.join(_dir, 'img1', img_prefix.format(frame_id)), h, w, global_image)
            image_meta = build_image(os.path.join(_dir, img_prefix.format(frame_id)), h, w, global_image)
            json_label['images'].append(image_meta)
            for annotation in gts[frame_id]:
                [bbox_left, bbox_top, bbox_width, bbox_height, object_category] = annotation
                anno = build_annotation(bbox_left, bbox_top, bbox_width, bbox_height, global_image, global_anno,
                                        object_category)
                json_label['annotations'].append(anno)
                global_anno += 1
            if frame_id in gts_ignore.keys():
                for annotation in gts_ignore[frame_id]:
                    [bbox_left, bbox_top, bbox_width, bbox_height, object_category] = annotation
                    anno = build_annotation(bbox_left, bbox_top, bbox_width, bbox_height, global_image, global_anno,
                                            0, ignore=1)
                    json_label['annotations'].append(anno)
                    global_anno += 1
            global_image += 1
            # print(f"{frame_id} done")

        # instead of writing to one file, write to one file per movie 
        # output_path = os.path.join()

    with open(args.output_anno, 'w') as f:
        json.dump(json_label, f)


if __name__=='__main__':
    main()
