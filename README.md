# Simple Stupid Online and Real-time object Tracking

Sometimes, [SORT](https://arxiv.org/abs/1602.00763) with Kalman filter isn't simple, lightweight and fast enough. Here we have SSORT.

## Usage

```python
# init_step: Minimum age of a track to be considered confirmed. At this stage, any misdetection will make the track be discarded
# min_hit: Minimum number of detection for a track. Tracks with smaller number of detection are ignored
# max_mis: Tracks with this or larger number of consecutive misdetection are discarded
# min_cost_threshold: Cost threshold for the most recent updated track
# max_cost_threshold: Cost threshold for the most out-of-date track
tracker = SSort(init_step=3, min_hit=5, max_mis=12, min_cost_threshold=0.66, max_cost_threshold=0.99)
while True:
    # Get your detections here
    # detections is a list of boxes, each box is in form [x1, y1, x2, y2, conf, class_id]
    detections = ...

    # Forward tracker's state to the next step
    tracker.forward()

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
