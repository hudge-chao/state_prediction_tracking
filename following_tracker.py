import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import OccupancyGrid
from state_track_network import state_predictor
import torch 
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

        self.local_map_resolution = 0.1

    def run(self):
        rospy.init_node('~', anonymous=True)

        self.leader_postion_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.leader_position_callback, queue_size=5)

        self.leader_map_sub = rospy.Subscriber('/occupancy_map_local', OccupancyGrid, self.leader_map_callback, queue_size=1)

        self.follower_navigation_pub = rospy.Publisher('/waypoint', PointStamped, queue_size=1)

        self.goal_msg_publish_seq = 0

        rospy.spin()


    def leader_position_callback(self, msg:ModelStates):
        leader_pose_index = msg.name.index('robot')
        leader_position = msg.pose[leader_pose_index].position
        self.leader_trajectoy.append(leader_position)
        if len(self.leader_trajectoy) > 35:
            self.leader_trajectoy.pop(0)
        
        if len(self.leader_trajectoy) == 35:
            # state tracker inference the leader state and publish the navigation goal waypoint
            tracked_state = self.inference_navigation_goal()
            self.publish_navigation_point(tracked_state)


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
        follower_local_map = self.get_follower_occupancy_map()
        follower_local_map.shape = (1, 1, self.myOccupancyMapHeight, self.myOccupancyMapWidth)
        follower_local_map_input = torch.from_numpy(follower_local_map).float().to(self.device)
        leader_trjectory_list = []
        for item in self.leader_trajectoy:
            X = int((item.x - self.myOccupancyMapOriginX) / 0.1)
            Y = int((item.y - self.myOccupancyMapOriginY) / 0.1)
            leader_trjectory_list.append([X, Y])
        leader_trjectory = np.array(leader_trjectory_list, dtype=np.float)
        leader_trjectory = leader_trjectory.reshape(-1)
        leader_trjectory = np.expand_dims(leader_trjectory, 0)
        leader_trjectory_input = torch.from_numpy(leader_trjectory).float().to(self.device)
        tracker_output = self.tracker_net(follower_local_map_input, leader_trjectory_input)
        track_position = tracker_output.detach().cpu().numpy()[0]
        return track_position

    def publish_navigation_point(self, navigation_goal):
        navigation_msg = PointStamped()
        navigation_msg.header.frame_id = 'map'
        navigation_msg.header.seq = self.goal_msg_publish_seq
        navigation_msg.header.stamp = rospy.Time.now()

        navigation_msg.point.x = navigation_goal[0]
        navigation_msg.point.y = navigation_goal[1]
        navigation_msg.point.z = 0.75

        self.goal_msg_publish_seq += 1

        self.follower_navigation_pub.publish(navigation_msg)

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

            


        