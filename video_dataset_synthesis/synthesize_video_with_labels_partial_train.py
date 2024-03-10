import os
import skvideo
import skvideo.io
import cv2
import numpy
import json
import joblib
import sys
import yaml
import torch
import json
import math
import torch.multiprocessing as mp
import ffmpeg
from lightning_training_model import DiffusionModel

frame_rate = 30
length = 40 # 40 seconds
images_per_second = 5
output_size = 256
channels = 3
jobs = 96
label_size = 9
t_length = length * images_per_second
video_source_folder = "/work/cseos2g/papapalpi/data/deepDrive/bdd100k/videos/train_smaller"
json_source_folder = "/work/cseos2g/papapalpi/data/deepDrive/bdd100k/info/100k/train"
output_file_frames = "./video_train_3x256x256.npy"
output_file_labels = "./video_train_label.npy"

def interpolate(a, target_length):
    current_length = a.shape[0]
    output = numpy.zeros(target_length, dtype=float)
    if (current_length == 0):
        print("TOO SHORT!")
        return output
    step_size = current_length / target_length
    current_range = numpy.arange(0, current_length, 1.0)
    target_range = numpy.arange(0, current_length, step_size)
    tmp = numpy.interp(target_range, current_range, a)[:target_length]
    output[:tmp.shape[0]] = tmp
    return output

def generate_labels(json_data):
    accel_data = json_data["accelerometer"]
    gyro_data = json_data["gyro"]
    gps_data = json_data["locations"]
    accel_y = interpolate(numpy.array([accel_data[i]["y"] for i in range(len(accel_data))], dtype=float), t_length).reshape(t_length, 1)
    accel_z = interpolate(numpy.array([accel_data[i]["z"] for i in range(len(accel_data))], dtype=float), t_length).reshape(t_length, 1)
    accel_x = interpolate(numpy.array([accel_data[i]["x"] for i in range(len(accel_data))], dtype=float), t_length).reshape(t_length, 1)
    gyro_y = interpolate(numpy.array([gyro_data[i]["y"] for i in range(len(gyro_data))], dtype=float), t_length).reshape(t_length, 1)
    gyro_z = interpolate(numpy.array([gyro_data[i]["z"] for i in range(len(gyro_data))], dtype=float), t_length).reshape(t_length, 1)
    gyro_x = interpolate(numpy.array([gyro_data[i]["x"] for i in range(len(gyro_data))], dtype=float), t_length).reshape(t_length, 1)
    gps_lat = interpolate(numpy.array([gps_data[i]["latitude"] for i in range(len(gps_data))], dtype=float), t_length).reshape(t_length, 1)
    gps_long = interpolate(numpy.array([gps_data[i]["longitude"] for i in range(len(gps_data))], dtype=float), t_length).reshape(t_length, 1)
    gps_speed = interpolate(numpy.array([gps_data[i]["speed"] for i in range(len(gps_data))], dtype=float), t_length).reshape(t_length, 1)

    return numpy.concatenate((accel_y, accel_z, accel_x, gyro_y, gyro_z, gyro_x, gps_lat, gps_long, gps_speed), axis=1)

def flushBuffer(frames, labels, mmap_frames_out, mmap_label_out, i, frame_insertion_location):
    #frames = torch.permute(torch.clamp(((frames / 255.0) - 0.5) * 2.0, -1.0, 1.0), (0, 3, 1, 2)).reshape(-1, 3, 256, 256)
    frames = torch.permute(frames, (0, 3, 1, 2)).reshape(-1, 3, 256, 256)
    mmap_frames_out[i][frame_insertion_location:(frame_insertion_location + frames.shape[0])] = frames.numpy()
    mmap_label_out[i][frame_insertion_location:(frame_insertion_location + labels.shape[0])] = labels.numpy()

def parseFile(file, json_file, i, mmap_frames_out, mmap_label_out):
    out, err = ffmpeg.input(file).output('pipe:', format='rawvideo', pix_fmt='rgb24', loglevel='quiet').run(capture_stdout=True)
    video = torch.from_numpy(numpy.frombuffer(out, numpy.uint8).reshape(-1, output_size, output_size, 3).copy())
    video = video[:t_length]
    json_file = json.load(open(json_file))
    print(json_file.keys())
    labels = torch.from_numpy(generate_labels(json_file)).type(torch.FloatTensor)
    flushBuffer(video, labels, mmap_frames_out, mmap_label_out, i, 0)
    print(f"PARSE DONE: {i}")
    sys.stdout.flush()

def parseFiles(video_files, sl):
    print(f"FILE PARSE {sl}!")
    sys.stdout.flush()
    output_memmap_frames = numpy.lib.format.open_memmap(output_file_frames, dtype=numpy.uint8, shape=(len(video_files), t_length, 3, output_size, output_size), mode='r+')
    output_memmap_labels = numpy.lib.format.open_memmap(output_file_labels, dtype=float, shape=(len(video_files), t_length, label_size), mode='r+')
    for i in range(sl.start, sl.stop):
        if (i >= len(video_files)):
            break
        tensor = parseFile(video_files[i][0], video_files[i][2], i, output_memmap_frames, output_memmap_labels)
        #q_output.put((tensor, i, 0))
    print(f"PARSE DONE {sl}!")
    sys.stdout.flush()

if __name__ == "__main__":
    video_files = [(os.path.join(video_source_folder, f), f, os.path.join(json_source_folder, f).replace(".mov", ".json")) for f in os.listdir(video_source_folder) if ".mov" in f]
    slice_size = math.ceil(len(video_files) / jobs)
    video_slices = [slice(video_start, video_start + slice_size) for video_start in range(0, len(video_files), slice_size)]
    print("TOTAL SLICES:", len(video_slices))

    output_memmap_frames = numpy.lib.format.open_memmap(output_file_frames, dtype=numpy.uint8, shape=(len(video_files), t_length, 3, output_size, output_size), mode='w+')
    output_memmap_labels = numpy.lib.format.open_memmap(output_file_labels, dtype=float, shape=(len(video_files), t_length, label_size), mode='w+')
    output_memmap_frames.flush()
    output_memmap_labels.flush()

    def runItAll():
        file_processes = []
        mp.set_start_method("forkserver")
        print("Starting new slices!")
        for sl in video_slices:
            print(sl)
            p = mp.Process(target=parseFiles, args=(video_files, sl))
            p.start()
            file_processes.append(p)
        for p in file_processes:
            p.join()
    runItAll()
