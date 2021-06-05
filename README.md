# Simple Stupid Online and Real-time object Tracking

Sometimes, [SORT](https://arxiv.org/abs/1602.00763) with Kalman filter isn't simple, lightweight and fast enough. Here we have SSORT.

## Usage

```
tracker = SSort(init_step=3, min_hit=5, max_mis=15, cost_threshold=4)
while True:
    # Get your detections here
    # detections is a list of boxes, each box is in form [x1, y1, x2, y2, conf, class_id]
    detections = ...

    # Forward tracker's state to the next step
    tracker.forward()

    # Match tracker's internal state with observations
    # observed_tracks: confirmed tracks that are matched with detections
    # deleted_tracks: confirmed tracks that are deleted
    # tracks_id: track id matched with each detection (confirmed or not does matter)
    observed_tracks, deleted_tracks, tracks_id = tracker.update(detections)
    
````

## Reference

[https://github.com/abewley/sort](https://github.com/abewley/sort)

[https://arxiv.org/pdf/1602.00763.pdf](https://arxiv.org/pdf/1602.00763.pdf)
