# Simple Stupid Online and Real-time object Tracking

Sometimes, [SORT](https://arxiv.org/abs/1602.00763) with Kalman filter isn't simple, lightweight and fast enough. Here we have SSORT.

## Usage

```python
# init_step: Minimum age of a track to be considered confirmed. At early stage, any misdetection will make this track be deleted
# max_mis: Tracks with this or larger number of consecutive misdetection are deleted
# min_iou: Minimum IOU between detection box and track's internal state box
# padding_ratio: Box are enlarged to have better IOU
# You MUST change these parameters to fit your problem
tracker = SSort(init_step=3, max_mis=10, min_iou=0.1, padding_ratio=0.25)
while True:
    # Get your detections here
    # detections is a list of boxes, each box is in form [x1, y1, x2, y2, conf, class_id]
    detections = ...

    # Match tracker's internal state with observations
    # observed_tracks: confirmed tracks that are matched with detections
    # deleted_tracks: confirmed tracks that are deleted
    # tracks_id: track id matched with each detection (confirmed or not doesn't matter)
    observed_tracks, deleted_tracks, tracks_id = tracker.update(detections)
    
```

## Run the demo

### Install dependencies

`pip install -r requirements.txt`

### Run the demo

`python demo.py <path_to_video>`

## Reference

[https://github.com/abewley/sort](https://github.com/abewley/sort)

[https://arxiv.org/pdf/1602.00763.pdf](https://arxiv.org/pdf/1602.00763.pdf)
