import os
import sys

import cv2
import yolov5

from tracker import SSort

DEMO_OUTPUT = 'demo_output'
SAVE_OUTPUT = False


def save_frame(frame):
    if not hasattr(save_frame, 'frame_count'):
        save_frame.frame_count = 0
    if (not os.path.exists(DEMO_OUTPUT)) or (not os.path.isdir(DEMO_OUTPUT)):
        os.makedirs(DEMO_OUTPUT)
    out_file = os.path.join(DEMO_OUTPUT, f'{save_frame.frame_count:05d}.jpg')
    cv2.imwrite(out_file, frame)
    save_frame.frame_count += 1


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python demo.py <path_to_video>')
        exit(1)

    video_path = sys.argv[1]
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print('Unable to open', video_path)
        exit(2)

    model = yolov5.load('yolov5s.pt',device='cuda:0')
    model.conf = 0.33
    model.iou = 0.45
    model.agnostic = True

    # You must change these parameters to fit your problem
    tracker = SSort(init_step=3,
                    max_mis=10,
                    min_iou=0.1,
                    padding_ratio=0.25)

    while True:
        ok, frame = video.read()
        if not ok or frame is None:
            break
        detections = model(frame)
        detections = detections.pred[0]
        detections = detections.tolist()
        observed_tracks, deleted_tracks, tracks_id = tracker.update(detections)
        for tid, detection in zip(tracks_id, detections):
            x1, y1, x2, y2 = [int(v) for v in detection[:4]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, str(tid), (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        if SAVE_OUTPUT:
            save_frame(frame)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1000 // 30) == 27:
            break
