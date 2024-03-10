import os
import skvideo
import skvideo.io
import cv2
import numpy
import joblib

video_source_folder = "/work/cseos2g/papapalpi/data/videos/bdd100k/videos/val"
video_files = [os.path.join(video_source_folder, f) for f in os.listdir(video_source_folder) if ".mov" in f]
frame_rate = 30
length = 40 # 40 seconds
images_per_second = 2
output_size = 256
channels = 3
jobs = 50
slice_size = 50
video_slices = [slice(video_start, video_start + slice_size) for video_start in range(0, len(video_files), slice_size)]

output_file = "./video_val_256x256.npy"

output_memmap = numpy.lib.format.open_memmap(output_file, dtype=numpy.uint8, shape=(len(video_files), length * images_per_second, channels, output_size, output_size), mode='w+')

def parseFile(file, mmap_out, i):
    reader = skvideo.io.vreader(file)
    frame_count = 0
    frame_insertion_location = 0
    for frame in reader:
        frame_sec_index = frame_count % frame_rate
        step_through_amount = frame_rate / images_per_second
        beginning_index = (frame_rate / 2) - (step_through_amount / 2)
        target_indexes = [int(beginning_index + (i * step_through_amount)) for i in range(images_per_second)]
        if (frame_sec_index in target_indexes):
            if (frame_insertion_location >= (length * images_per_second)):
                break
            frame = cv2.resize(frame, (output_size, output_size))
            frame = numpy.transpose(frame, (2, 0, 1))
            mmap_out[i][frame_insertion_location] = frame
            frame_insertion_location = frame_insertion_location + 1
        frame_count = frame_count + 1
    print(i)
        #cv2.imshow("image", frame)
        #cv2.waitKey(0)

def parseFiles(mmap_out, sl):
    for i in range(sl.start, sl.stop):
        parseFile(video_files[i], mmap_out, i)

def runItAll():
    joblib.Parallel(n_jobs=jobs)(joblib.delayed(parseFiles)(output_memmap, sl)
                   for sl in video_slices)

runItAll()
