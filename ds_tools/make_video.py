"""Overlay frames -> mp4 (cv2 MJPG .avi; the shipped .mp4 files were re-encoded to H.264 with ffmpeg). python make_video.py <overlay_dir> <out.mp4> <fps>"""
import sys, glob, cv2
fr = sorted(glob.glob(sys.argv[1] + "/*.jpg")); fps = float(sys.argv[3])
im = cv2.imread(fr[0]); h, w = im.shape[:2]
vw = cv2.VideoWriter(sys.argv[2], cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
for f in fr: vw.write(cv2.imread(f))
vw.release(); print("video", sys.argv[2], len(fr), "frames")
