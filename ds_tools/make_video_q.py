"""Overlay frames -> MJPG .avi, optionally downscaled to stay < 20 MB (this OpenCV build has no mp4v/H.264 writer). python make_video_q.py <overlay_dir> <out.avi> <fps> [scale=1.0]"""
import sys, glob, cv2
fr = sorted(glob.glob(sys.argv[1] + "/*.jpg")); fps = float(sys.argv[3]); s = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
im = cv2.imread(fr[0]); h, w = im.shape[:2]; W, H = int(w * s) // 2 * 2, int(h * s) // 2 * 2
vw = cv2.VideoWriter(sys.argv[2], cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))
for f in fr: vw.write(cv2.resize(cv2.imread(f), (W, H), interpolation=cv2.INTER_AREA))
vw.release(); print("video", sys.argv[2], len(fr), "frames", W, "x", H)
