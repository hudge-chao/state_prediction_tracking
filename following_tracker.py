from re import L
import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import OccupancyGrid
import torch 
from state_track_network import state_predictor
import threading
import cv2
import numpy as np

class StateTrackerFollow(threading.Thread):
    def __init__(self) -> None:
        super(StateTrackerFollow, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tracker_net = state_predictor()

        self.tracker_net.load_state_dict(torch.load('./weights/state_tracker/300_predictor.pth'), strict=True).to(self.device)
        
        self.leader_trajectoy = []

        self.follower_occupancy_map = []

    def run(self):
        rospy.init_node('~', anonymous=True)

        leader_postion_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.leader_position_callback, queue_size=5)

        leader_map_sub = rospy.Subscriber('/occupancy_map_local', OccupancyGrid, self.leader_map_callback, queue_size=1)

        follower_navigation_pub = rospy.Publisher('/waypoint', PointStamped, queue_size=1)

        rospy.spin()


    def leader_position_callback(self, msg:ModelStates):
        leader_pose_index = msg.name.index('robot')
        leader_position = msg.pose[leader_pose_index].position
        self.leader_trajectoy.append(leader_position)
        if len(self.leader_trajectoy) >= 35:
            self.leader_trajectoy.pop(0)

    def leader_map_callback(self, map:OccupancyGrid):
        self.myOccupancyMapWidth = map.info.width
        self.myOccupancyMapHeight = map.info.height
        self.myOccupancyMapOriginX = map.info.origin.position.x
        self.myOccupancyMapOriginY = map.info.origin.position.y
        
        occupancy_map_size = self.myOccupancyMapWidth * self.myOccupancyMapHeight

        if occupancy_map_size == 0:
            print('get error msg')
            return

        self.follower_occupancy_map = [map.data[index] for index in range(occupancy_map_size)]


    def inference_navigation_goal(self):
        
        pass

    def publish_navigation_point(self, navigation_goal):
        self.follower_navigation_pub.publish(navigation_goal)

    def get_follower_occupancy_map(self):
        
        occupancy_map = np.zeros((self.myOccupancyMapHeight, self.myOccupancyMapWidth), dtype=np.uint8)

        for row in range(self.myOccupancyMapHeight):
            for col in range(self.myOccupancyMapWidth):
                index = row * self.myOccupancyMapWidth + col
                data = self.follower_occupancy_map[index]
                if data == -1:
                    occupancy_map[row][col] = 100
                elif data == 0:
                    occupancy_map[row][col] = 255
                else:
                    occupancy_map[row][col] = 0
        
        return occupancy_map

            


        