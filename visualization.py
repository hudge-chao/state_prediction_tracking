import sys
from time import sleep
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import random 


def visualization_tools(img, path, target_position, prediction=None):
    # format input to right type
    img = img.astype(np.uint8)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    path = path.astype(np.int)
    target_position = target_position.astype(np.int)
    target_position = np.array([target_position[1], target_position[0]]) 
    # visualization  
    cv2.circle(img, (100, 100), 2, (0, 255, 0), 1)
    point_nums = len(path)
    for i in range(point_nums-1):
        start = (path[i][1], path[i][0])
        end = (path[i+1][1], path[i+1][0])
        cv2.line(img, start, end, (255, 0 , 0), 1)
    cv2.circle(img, target_position, 3, (0, 0, 255), 1)
    if prediction is not None:
        prediction = prediction.astype(np.int)
        prediction = np.array([prediction[1], prediction[0]]) 
        cv2.circle(img, prediction, 3, (255, 0, 255), 1)
    return img


def test_one_case(index:int):
    local_map = cv2.imread("samples/maps/{}.png".format(index))
    # origin_point = np.loadtxt("samples/origin/6798.csv", delimiter=',', dtype=np.float)
    # tracked_point = [int((tracked_position[0] - origin_point[0]) / 0.1), int((tracked_position[1] - origin_point[1]) / 0.1)]
    trajectory = np.loadtxt("samples/path/{}.csv".format(index), delimiter=',', dtype=np.int)
    tracked_position = np.loadtxt("samples/tracked_pixel/{}.csv".format(index), delimiter=',', dtype=np.int)
    visualized_image = visualization_tools(local_map, trajectory, tracked_position)
    cv2.imwrite("visualization_test.png", visualized_image)

def main():
    for _ in range(100):
        index = random.randint(0, 3000)
        test_one_case(index)
        sleep(2)

if __name__ == '__main__':
    main()