import argparse

import toml

from dataset.lavdf import LAVDFDataModule
from inference import inference_batfd
from metrics import AP, AR
from .model/batfd_MA import BATFD
from post_process import post_process
from utils import read_json

parser = argparse.ArgumentParser(description="BATFD evaluation")
parser.add_argument("--config", type=str)
parser.add_argument("--data_root", type=str)
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--gpus", type=int, default=1)

if __name__ == '__main__':
    args = parser.parse_args()
    config = toml.load(args.config)
    model_name = config["name"]
    alpha = config["soft_nms"]["alpha"]
    t1 = config["soft_nms"]["t1"]
    t2 = config["soft_nms"]["t2"]

    # prepare dataset
    dm = LAVDFDataModule(root=args.data_root,
        frame_padding=config["num_frames"],
        max_duration=config["max_duration"],
        batch_size=1, num_workers=4,
        get_meta_attr=BATFD.get_meta_attr)
    dm.setup()

    # prepare model
    model = BATFD.load_from_checkpoint(args.checkpoint)

    # inference and save dense proposals as csv file
    inference_batfd(model_name, model, dm, config["max_duration"], args.gpus)

    # postprocess by soft-nms
    metadata = dm.test_dataset.metadata
    post_process(model_name, metadata, 25, alpha, t1, t2)
    proposals = read_json(f"output/results/{model_name}.json")

    # evaluate AP
    iou_thresholds = [0.5, 0.75, 0.95]
    print("--------------------------------------------------")
    ap_score = AP(iou_thresholds=iou_thresholds)(metadata, proposals)
    for iou_threshold in iou_thresholds:
        print(f"AP@{iou_threshold} Score for all modality in full set: {ap_score[iou_threshold]}")
    print("--------------------------------------------------")

    # evaluate AR
    iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    n_proposals_list = [100, 50, 20, 10]

    ar_score = AR(n_proposals_list, iou_thresholds=iou_thresholds, parallel=False)(metadata, proposals)

    for n_proposals in n_proposals_list:
        print(f"AR@{n_proposals} Score for all modality in full set: {ar_score[n_proposals]}")
    print("--------------------------------------------------")
