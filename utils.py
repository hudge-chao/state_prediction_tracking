import functools
import os
import numpy as np
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path: 
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from tqdm import tqdm, trange
sys.path.append(os.getcwd())

def file_cmp(f1: str, f2: str):
    seq_f1 = int(f1.split('.')[0])
    seq_f2 = int(f2.split('.')[0])
    if seq_f1 < seq_f2:
        return -1
    elif seq_f1 > seq_f2:
        return 1
    else:
        return 0

# path = 'samples/localmaps'
# path = 'localmaps'

def tracked_to_prob(tracked_position, input_image : np.ndarray) -> np.ndarray:    
    prob_mask = [0.50, 0.55, 0.65, 0.70, 0.65, 0.55, 0.50,
                 0.55, 0.65, 0.75, 0.80, 0.75, 0.65, 0.55,
                 0.65, 0.75, 0.85, 0.90, 0.85, 0.75, 0.65,
                 0.70, 0.80, 0.90, 1.00, 0.90, 0.80, 0.70,
                 0.65, 0.75, 0.85, 0.90, 0.85, 0.75, 0.65,
                 0.55, 0.65, 0.75, 0.80, 0.75, 0.65, 0.55,
                 0.50, 0.55, 0.65, 0.70, 0.65, 0.55, 0.50]
    
    prob_mask_graph = np.array(prob_mask, dtype=np.float).reshape(7, 7)
    
    # array = [0.1 for i in range(90000)]

    # prob_graph = np.array(array, np.float).reshape(300, 300)

    prob_graph = np.zeros((300, 300), dtype=np.float)

    for y in range(int(tracked_position[0] - 3), int(tracked_position[0] + 3)):
        for x in range(int(tracked_position[1] - 3), int(tracked_position[1] + 3)):
            if x < 0 or x > 300 or y < 0 or y > 300:
                continue
            prob_graph[y][x] = prob_mask_graph[int(y - tracked_position[0] + 3)][int(x - tracked_position[1] + 3)]

    for row in range(300):
        for col in range(300):
            prob = prob_graph[row][col]
            prob_graph[row][col] = (-1.0 if input_image[row][col] == 0 else prob)
    
    return prob_graph


tracked_path = 'samples/tracked_pixel'

map_path = 'samples/maps'

tracked_dst = os.path.join(os.getcwd(), tracked_path)
map_dst = os.path.join(os.getcwd(), map_path)
output_dst = os.path.join(os.getcwd(), 'samples/tracked_prob')

# map = cv2.imread('samples/maps/0.png', cv2.IMREAD_GRAYSCALE)
# tracked_position = np.loadtxt('samples/tracked_pixel/0.csv', delimiter=',', dtype=np.int)
# graph = tracked_to_prob(tracked_position, map)
# cv2.imwrite('prob.png', graph * 255)

dirs = os.listdir(tracked_dst)

pbar = tqdm(total=len(dirs))

for i, dir in enumerate(dirs):
    tracked_file = os.path.join(tracked_path, dir)
    map_file = os.path.join(map_dst, f'{i}.png')
    map = cv2.imread(map_file, cv2.IMREAD_GRAYSCALE)
    tracked_position = np.loadtxt(tracked_file, delimiter=',', dtype=np.int)
    tracked_prob = tracked_to_prob(tracked_position, map)
    output_path = os.path.join(output_dst, f'{i}.png')
    cv2.imwrite(output_path, tracked_prob)
    pbar.update(1)

    

# for dir in dirs:
#     parent_path = os.path.join(dst, dir)
#     print(parent_path)
#     files = os.listdir(parent_path)
#     file_type = files[0].split('.')[1]
#     print(file_type)
#     files.sort(key= functools.cmp_to_key(file_cmp))
#     i = 0
#     for file in files:
#         old_file_name = parent_path + os.sep + file
#         new_file_name = parent_path + os.sep + str(i) + '.' + file_type
#         print(new_file_name)
#         os.rename(old_file_name, new_file_name)
#         print(old_file_name, '======>', new_file_name)
#         i += 1

# files.sort(key= functools.cmp_to_key(file_cmp))

# print(len(files))
# print(files[0])
# print(int(files[0].split('.')[0]))

# i = 0
# for file in files:
#     old_file_name = dst + os.sep + file
#     new_file_name = dst + os.sep + str(i)+'.png'
#     os.rename(old_file_name, new_file_name)
#     print(old_file_name, '======>', new_file_name)
#     i += 1
