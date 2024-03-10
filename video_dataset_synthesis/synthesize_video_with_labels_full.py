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
from lightning_training_model import DiffusionModel

frame_rate = 30
length = 40 # 40 seconds
images_per_second = 5
output_size = 256
channels = 3
jobs = 24
gpu_jobs = 2
latent_size = 512
label_size = 9
batch_size = 100
t_length = length * images_per_second

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

def generate_default_values(keys, t_length):
    empty = [0.0 for i in range(t_length)]
    result = {}
    for key in keys:
        result[key] = empty
    return result

def generate_labels(json_data):
    accel_data = json_data["accelerometer"] if "accelerometer" in json_data else generate_default_values(["y", "z", "x"], t_length)
    gyro_data = json_data["gyro"] if "gyro" in json_data else generate_default_values(["y", "z", "x"], t_length)
    gps_data = json_data["locations"] if "locations" in json_data else generate_default_values(["latitude", "longitude", "speed"], t_length)
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

def parseFile(file, hash, json_file, i, q_output):
    reader = skvideo.io.vreader(file)
    json_file = json.load(open(json_file))
    print(json_file.keys())
    labels = torch.from_numpy(generate_labels(json_file)).type(torch.FloatTensor)
    frame_count = 0
    frame_insertion_location = 0
    current_queue_frames = []
    current_queue_labels = []
    step_through_amount = frame_rate / images_per_second
    beginning_index = 0
    target_indexes = [int(beginning_index + (i * step_through_amount)) for i in range(images_per_second)]
    print("INDEXES: ", len(target_indexes), target_indexes)
    for frame in reader:
        frame_sec_index = frame_count % frame_rate
        #if (frame_sec_index in target_indexes):
        if (frame_count >= t_length):
            print(f"BREAKING FINAL INSERTION LOCATION: {frame_insertion_location}, TOTAL FRAMES: {frame_count}, {i}")
            break
        #frame = cv2.resize(frame, (output_size, output_size))
        frame = torch.from_numpy(frame.copy()).type(torch.FloatTensor)
        #frame = numpy.transpose(frame, (2, 0, 1))
        #frame = torch.clamp(((torch.from_numpy(frame.copy()).type(torch.FloatTensor) / 255.0) - 0.5) * 2.0, -1.0, 1.0).reshape(1, 3, 256, 256)
        current_labels = labels[frame_count].reshape(1, label_size)
        current_queue_frames.append(frame)
        current_queue_labels.append(current_labels)
        frame_count = frame_count + 1
        if (len(current_queue_frames) >= batch_size):
            current_queue_frames = torch.permute(torch.clamp(((torch.cat(current_queue_frames) / 255.0) - 0.5) * 2.0, -1.0, 1.0), (2, 0, 1)).reshape(-1, 3, 256, 256)
            current_queue_labels = torch.cat(current_queue_labels)
            current_queue_frames.share_memory_()
            current_queue_labels.share_memory_()
            q_output.put((current_queue_frames, current_queue_labels, i, frame_insertion_location))
            current_queue_frames = []
            current_queue_labels = []
            frame_insertion_location = frame_insertion_location + batch_size
    if (len(current_queue_frames) > 0):
        current_queue_frames = torch.permute(torch.clamp(((torch.cat(current_queue_frames) / 255.0) - 0.5) * 2.0, -1.0, 1.0), (2, 0, 1)).reshape(-1, 3, 256, 256)
        current_queue_labels = torch.cat(current_queue_labels)
        current_queue_frames.share_memory_()
        current_queue_labels.share_memory_()
        q_output.put((current_queue_frames, current_queue_labels, i, frame_insertion_location))
    print(f"FINAL INSERTION LOCATION: {frame_insertion_location}, TOTAL FRAMES: {frame_count}, {i}")
    print(f"PARSE DONE: {i}")
    sys.stdout.flush()
        #cv2.imshow("image", frame)
        #cv2.waitKey(0)

def gpuThread(i, q_input, q_output):
    torch.backends.cudnn.benchmark = True
    with torch.no_grad():
        with torch.autocast("cuda"):
            path = "/work/cseos2g/papapalpi/data/models/DashcamDiffusionAutoencoder64x64New/saved_diffusion_models/our_largest_model/model.ckpt"
            config_path = "./config_files/diffusion_model_64x64x3_largest_model.yaml"
            config = yaml.safe_load(open(config_path))
            device = f"cuda:{i}"
            model = DiffusionModel.load_from_checkpoint(path, config=config).eval().to(device)
            count = 1
            total = 0
            while (count > 0):
                frames, labels, i, frame_insertion_location = q_input.get()
                if not torch.is_tensor(frames):
                    print("GPU EXITING!")
                    count = count - 1
                else:
                    frames = frames.to(device)
                    latent = model.encode_semantic(frames).cpu().detach()
                    #print("Latent shape:", latent.shape, "Label shape:", labels.shape, i, frame_insertion_location)
                    q_output.put((latent, labels, i, frame_insertion_location))
                print(f"GPU: {total}")
                if ((total % 100) == 0):
                    sys.stdout.flush()
                total = total + 1
                #q_input.task_done()
            q_output.put((False, False, False, False))
    return


def mmapThread(q_input, mmap_latent_out, mmap_label_out):
    count = 1
    print("MMAP THREAD LAUNCHING!")
    while (count > 0):
        latent, labels, i, frame_insertion_location = q_input.get()
        if not torch.is_tensor(latent):
            print("EXITING!")
            print(latent, i, frame_insertion_location)
            count = 0
        else:
            mmap_latent_out[i][frame_insertion_location:(frame_insertion_location + latent.shape[0])] = latent.numpy()
            mmap_label_out[i][frame_insertion_location:(frame_insertion_location + labels.shape[0])] = labels.numpy()
            mmap_latent_out.flush()
            mmap_label_out.flush()
        #q_input.task_done()
    return

def parseFiles(video_files, q_output, sl):
    print(f"FILE PARSE {sl}!")
    sys.stdout.flush()
    for i in range(sl.start, sl.stop):
        if (i >= len(video_files)):
            break
        tensor = parseFile(video_files[i][0], video_files[i][1], video_files[i][2], i, q_output)
        #q_output.put((tensor, i, 0))
    print(f"PARSE DONE {sl}!")
    sys.stdout.flush()

if __name__ == "__main__":
    video_source_folder = "/work/cseos2g/papapalpi/data/deepDrive/bdd100k/videos/val_smaller"
    json_source_folder = "/work/cseos2g/papapalpi/data/deepDrive/bdd100k/info/100k/val"
    video_files = [(os.path.join(video_source_folder, f), f, os.path.join(json_source_folder, f).replace(".mov", ".json")) for f in os.listdir(video_source_folder) if ".mov" in f]
    video_files = video_files
    slice_size = math.ceil(len(video_files) / jobs)
    video_slices = [slice(video_start, video_start + slice_size) for video_start in range(0, len(video_files), slice_size)]
    print("TOTAL SLICES:", len(video_slices))

    output_file_latent = "./video_val_512.npy"
    output_file_labels = "./video_val_label.npy"

    output_memmap_latent = numpy.lib.format.open_memmap(output_file_latent, dtype=float, shape=(len(video_files), t_length, latent_size), mode='w+')
    output_memmap_labels = numpy.lib.format.open_memmap(output_file_labels, dtype=float, shape=(len(video_files), t_length, label_size), mode='w+')

    def runItAll():
        file_processes = []
        gpu_processes = []
        mmap_processes = []
        mp.set_start_method("forkserver")
        print("Starting new slices!")
        gpuQueue = mp.Queue(10000)
        mmapQueue = mp.Queue(10000)
        for sl in video_slices:
            print(sl)
            p = mp.Process(target=parseFiles, args=(video_files, gpuQueue, sl))
            p.start()
            file_processes.append(p)
            p = mp.Process(target=mmapThread, args=(mmapQueue, output_memmap_latent, output_memmap_labels))
            p.start()
            mmap_processes.append(p)
        for i in range(gpu_jobs):
            p = mp.Process(target=gpuThread, args=(i, gpuQueue, mmapQueue))
            p.start()
            gpu_processes.append(p)
        for p in file_processes:
            p.join()
        for i in range(gpu_jobs):
            gpuQueue.put((False, False, False, False))
        for p in gpu_processes:
            p.join()
        for i in range(jobs):
            mmapQueue.put((False, False, False, False))
        for p in mmap_processes:
            p.join()
        #joblib.Parallel(n_jobs=jobs)(joblib.delayed(parseFiles)(model, output_memmap_latent, sl)
        #               for sl in video_slices)
    runItAll()
