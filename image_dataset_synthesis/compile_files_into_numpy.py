import numpy
from PIL import Image
import os
import os.path
import joblib
import sys
import json
import cv2

name = sys.argv[1]
output_image_memmap = name + "_images.npy" #"train.npy"
output_label_memmap = name + "_labels.npy"
folder = sys.argv[2]
json_file = sys.argv[3]
slice_size = 1000
jobs = 20
paths = [e for e in os.listdir(folder) if "jpg" in e]
image_count = len(paths)
image_slices = [slice(start, start + slice_size) for start in range(0, image_count - slice_size, slice_size)]
print(name, output_image_memmap, output_label_memmap, folder, json_file)

json_raw_data = json.load(open(json_file))
json_data = {}
for entry in json_raw_data:
    json_data[entry['name']] = entry['attributes']

del json_raw_data

# Generate our 3 entry numpy array that respectively creates:
# Daytime (1) or nighttime (0)
# highway (1) vs not (0)
# Clouds (1) or no clouds  (0)
def generate_attribute_data(json_entry):
    daytime = 1 if (json_entry['timeofday'] == 'daytime') else 0
    highway = 1 if (json_entry['scene'] == 'highway') else 0
    clouds = 1 if (json_entry['weather'] != 'clear') else 0
    result = numpy.array([daytime, highway, clouds], dtype=numpy.uint8)
    return result

def exportToNumpy(memmap_array_image, memmap_array_labels, our_paths, sl):
    start = sl.start
    stop = sl.stop
    print("Job launched!")
    for i in range(start, stop):
        #print(i, i - start)
        image_path = our_paths[i - start]
        final_image_path = os.path.join(folder, image_path)
        image = numpy.array(Image.open(final_image_path), dtype=numpy.uint8)
        image_downscaled = cv2.resize(image, dsize=(256,256), interpolation=cv2.INTER_CUBIC)
        json_entry = json_data[image_path] if (image_path in json_data) else {'timeofday': 'not_daytime', 'scene': 'not_highway', 'weather': 'notclear'}
        if ((image_path not in json_data)):
            print("WTF?!")
        memmap_array_image[i] = numpy.array(image_downscaled, dtype=numpy.uint8).transpose((2,0,1))
        memmap_array_labels[i] = generate_attribute_data(json_entry)

        #print(i, i - start)
        sys.stdout.flush()

def runItAll():
    print("Launching full run!")
    output_image = numpy.lib.format.open_memmap(output_image_memmap, dtype=numpy.uint8, shape=(image_count, 3, 256, 256), mode='w+')
    output_label = numpy.lib.format.open_memmap(output_label_memmap, dtype=numpy.uint8, shape=(image_count, 3), mode='w+')
    joblib.Parallel(n_jobs=jobs)(joblib.delayed(exportToNumpy)(output_image, output_label, paths[sl], sl)
                   for sl in image_slices)

runItAll()
