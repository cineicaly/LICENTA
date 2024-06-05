from engine.trackers.strongsort.utils.parser import get_config
from engine.trackers.strongsort.strong_sort import StrongSORT
from engine.trackers.ocsort.ocsort import OCSort
#from engine.trackers.bytetrack.byte_tracker import BYTETracker
from pathlib import Path


class MultiObjectTracking:
    def __init__(self):
        self.cfg = get_config()


    def ocsort(self, min_hits=None, max_age=None, iou_threshold=None):
        tracker_config = Path(r"engine/trackers/ocsort/configs/ocsort.yaml")
        self.cfg.merge_from_file(tracker_config)

        min_hits = min_hits if min_hits is not None else self.cfg.ocsort.min_hits
        max_age = max_age if max_age is not None else self.cfg.ocsort.max_age
        iou_threshold = iou_threshold if iou_threshold is not None else self.cfg.ocsort.iou_thresh

        ocsort = OCSort(
            det_thresh=self.cfg.ocsort.det_thresh,
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
            delta_t=self.cfg.ocsort.delta_t,
            asso_func=self.cfg.ocsort.asso_func,
            inertia=self.cfg.ocsort.inertia,
            use_byte=self.cfg.ocsort.use_byte,
        )
        return ocsort

    # def bytetrack(self):
    #     tracker_config = Path(r"engine/trackers/bytetrack/configs/bytetrack.yaml")
    #     self.cfg.merge_from_file(tracker_config)
    #
    #     bytetracker = BYTETracker(
    #         track_thresh=self.cfg.bytetrack.track_thresh,
    #         match_thresh=self.cfg.bytetrack.match_thresh,
    #         track_buffer=self.cfg.bytetrack.track_buffer,
    #         frame_rate=self.cfg.bytetrack.frame_rate
    #     )
    #     return bytetracker

    def strongsort(self, n_init=None, max_age=None, max_iou_dist=None, max_dist=None, device="cpu", half=False):
        reid_weights = Path(r"engine/trackers/strongsort/deep/checkpoint/osnet_x0_25_msmt17.pt")

        tracker_config = Path(r"engine/trackers/strongsort/configs/strongsort.yaml")
        self.cfg.merge_from_file(tracker_config)

        n_init = n_init if n_init is not None else self.cfg.strongsort.n_init
        max_age = max_age if max_age is not None else self.cfg.strongsort.max_age
        max_iou_dist = max_iou_dist if max_iou_dist is not None else self.cfg.strongsort.max_iou_dist
        max_dist = max_dist if max_dist is not None else self.cfg.strongsort.max_dist

        strongsort = StrongSORT(
            reid_weights,
            device,
            half,
            max_dist=max_dist,
            max_iou_dist=max_iou_dist,
            max_age=max_age,
            max_unmatched_preds=self.cfg.strongsort.max_unmatched_preds,
            n_init=n_init,
            nn_budget=self.cfg.strongsort.nn_budget,
            mc_lambda=self.cfg.strongsort.mc_lambda,
            ema_alpha=self.cfg.strongsort.ema_alpha,
        )
        return strongsort
