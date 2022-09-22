import math
import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PointStamped, Pose
from nav_msgs.msg import OccupancyGrid
from models.Tracker.state_track_network import state_predictor
from models.Tracker.state_track_trajectory_only import state_track_trajectory
from visualization_msgs.msg import Marker
from tf.broadcaster import TransformBroadcaster
from tf.transformations import *
import sys
import torch 
import threading
import cv2
import numpy as np

class TrackerEvaluate(threading.Thread):
    def __init__(self, local_map_resolution : float = 0.1, save_visulization_img : bool = False) -> None:
        super(TrackerEvaluate, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # tracker one
        self.tracker_union_net = state_predictor(model='deploy')

        self.tracker_union_net.load_state_dict(torch.load('./weights/state_tracker_union/tracker.pth'), strict=True)
    
        self.tracker_union_net.to(self.device)

        # tracker two
        self.tracker_single_net = state_track_trajectory(model='deploy')

        self.tracker_single_net.load_state_dict(torch.load('./weights/state_tracker_single/tracker.pth'), strict=True)

        self.tracker_single_net.to(self.device)

        # other variables
        
        self.leader_trajectoy = []

        self.leader_trajectory_dist = []

        self.follower_poistion = None

        self.follower_occupancy_map = []

        self.follower_local_map = []

        self.map_origin_array = []

        self.local_map_resolution = local_map_resolution

        self.save_visulization_img = save_visulization_img

        self.trajectory_waypoints_nums = 35

        self.tf_boardcaster = TransformBroadcaster()

        rospy.init_node('state_track_node')


    def run(self):
        self.leader_postion_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.leader_position_callback, queue_size=5)

        self.leader_map_sub = rospy.Subscriber('/occupancy_map_local', OccupancyGrid, self.leader_map_callback, queue_size=1)

        self.follower_navigation_pub = rospy.Publisher('/waypoint', PointStamped, queue_size=1)

        self.visulization_marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)

        self.goal_msg_publish_seq = 0

        rate = rospy.Rate(1)

        while not rospy.is_shutdown():
            if len(self.leader_trajectoy) == self.trajectory_waypoints_nums and len(self.follower_local_map) == 30:
                # state tracker inference the leader state and publish the navigation goal waypoint
                tracked_state_real_union, tracked_state_img, tracked_state_real_single = self.inference_navigation_goal()
                state_real = self.get_current_leader_position()
                print(tracked_state_real_union, "    ", state_real, "    ", tracked_state_real_single)
                state_msgs = [state_real, tracked_state_real_union, tracked_state_real_single]
                self.rviz_visulization_tools(state_msgs)
                file_output = open('record.csv', 'a')
                file_output.write('{},{},{},{},{},{}\n'.format(state_real[0], state_real[1], tracked_state_real_union[0], tracked_state_real_union[1], tracked_state_real_single[0], tracked_state_real_single[1]))
                file_output.close()
            rate.sleep()

    def leader_position_callback(self, msg:ModelStates):
        leader_pose_index = msg.name.index('robot')
        leader_position = msg.pose[leader_pose_index].position

        if len(self.follower_local_map) > 0:
            is_visiable = self.is_leader_in_follower_fov()

        # 跟随者丢失引领者位置
        if not is_visiable:
            return

        self.leader_trajectoy.append(leader_position)

        if len(self.leader_trajectory_dist) == 0:
            self.leader_trajectory_dist.append(leader_position)
            self.follower_poistion = leader_position    
        else:
            last_leader_position = self.leader_trajectory_dist[-1]

            dist = math.sqrt(pow(leader_position.x - last_leader_position.x, 2) + pow(leader_position.y - last_leader_position.y, 2))

            if dist >= 0.1:
                self.leader_trajectory_dist.append(leader_position)

        if len(self.leader_trajectory_dist) > 30:
            self.leader_trajectory_dist.pop(0)

        # 解决跟随者位置问题
        dist_follower_leader = math.sqrt(pow(self.follower_poistion.x - leader_position.x, 2) + pow(self.follower_poistion.y - leader_position.y, 2))

        if dist_follower_leader >= 3.0:
            self.follower_poistion = self.leader_trajectory_dist[0]
        
        self.boardcast_follower_transform()

        if len(self.leader_trajectoy) > self.trajectory_waypoints_nums:
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

        assert(occupancy_map_size > 0)

        self.follower_occupancy_map = [map.data[index] for index in range(occupancy_map_size)]
        occupancy_map = self.get_follower_occupancy_map()
        self.follower_local_map.append(occupancy_map)
        self.map_origin_array.append([self.myOccupancyMapOriginX, self.myOccupancyMapOriginY])
        if len(self.map_origin_array) > 30:
            self.follower_local_map.pop(0)
            self.map_origin_array.pop(0)

    def boardcast_follower_transform(self):
        translation = (self.follower_poistion.x, self.follower_poistion.y, 0.75)
        rotation = (0.0, 0.0, 0.0, 1.0)
        self.tf_boardcaster.sendTransform(translation, rotation, 
                                        rospy.Time.now(),
                                        'sim_vehicle',
                                        'map')

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

    def get_current_leader_position(self):
        return [self.leader_trajectoy[-1].x, self.leader_trajectoy[-1].y]

    def inference_navigation_goal(self):
        # prepare input data for tracker network to inference
        follower_local_map = self.follower_local_map[0]
        follower_local_map.shape = (1, 1, self.myOccupancyMapHeight, self.myOccupancyMapWidth)
        follower_local_map_input = torch.from_numpy(follower_local_map).float().to(self.device)
        leader_trajectory_img_list = []
        leader_trajectory_real_list = []
        mapOriginAlign = self.map_origin_array[0]
        for index, item in enumerate(self.leader_trajectoy):
            if index < 30:    
                X = int((item.x - mapOriginAlign[0]) / 0.1)
                Y = int((item.y - mapOriginAlign[1]) / 0.1)
                leader_trajectory_img_list.append([X, Y])
                leader_trajectory_real_list.append([item.x, item.y])
        # use union input tracker 
        leader_trajectory_img = np.array(leader_trajectory_img_list, dtype=np.float)
        leader_trajectory_img = leader_trajectory_img.reshape(-1)
        leader_trajectory_img = np.expand_dims(leader_trajectory_img, 0)
        leader_trajectory_input = torch.from_numpy(leader_trajectory_img).float().to(self.device)
        tracker_union_output = self.tracker_union_net(follower_local_map_input, leader_trajectory_input)
        # use single input tracker 
        leader_trajectory_real = np.array(leader_trajectory_real_list, dtype=np.float)
        leader_trajectory_real = leader_trajectory_real.reshape(-1)
        leader_trajectory_real = np.expand_dims(leader_trajectory_real, 0)
        leader_trajectory_input = torch.from_numpy(leader_trajectory_real).float().to(self.device)
        tracker_single_output = self.tracker_single_net(leader_trajectory_input)

        track_position_img_union = tracker_union_output.detach().cpu().numpy()[0]
        track_position_real_single = tracker_single_output.detach().cpu().numpy()[0]

        track_position_real_union = [track_position_img_union[0] * self.local_map_resolution + mapOriginAlign[0], track_position_img_union[1] * self.local_map_resolution + mapOriginAlign[1]]
        
        return track_position_real_union, track_position_img_union, track_position_real_single

    def rviz_visulization_tools(self, msgs):        
        markers = [Marker() for _ in msgs]
        for i, m in enumerate(markers):
            m.header.frame_id = '/markers'
            m.header.stamp = rospy.Time.now()
            m.ns = 'state_tracking'
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.MODIFY
            pose = Pose()
            pose.position.x = msgs[i][0]
            pose.position.y = msgs[i][1]
            pose.position.z = 0.75
            pose.orientation.w = 1.0
            m.pose = pose
            m.scale.x = 0.5
            m.scale.y = 0.5
            m.scale.z = 1.5
            if i == 0:
                m.color.r = 1.0
                m.color.g = 0.0
                m.color.b = 0.0
            elif i == 1:
                m.color.r = 0.0
                m.color.g = 1.0
                m.color.b = 0.0
            else:
                m.color.r = 0.0
                m.color.g = 0.0
                m.color.b = 1.0
            m.color.a = 1.0
            m.lifetime = rospy.Duration()
            self.visulization_marker_pub.publish(m)

    def is_leader_in_follower_fov(self) -> bool:
        leader_current_loc = self.get_current_leader_position()
        follower_map_origin = self.map_origin_array[0]
        leader_current_loc = [(leader_current_loc[0] - follower_map_origin[0])/0.1, (leader_current_loc[1] - follower_map_origin[1])/0.1]
        follower_current_map = self.follower_local_map[0]
        follower_fixed_loc = [150, 150]
        # if follower_fixed_loc[0] != leader_current_loc[0]:
        #     leader_to_follower_slope = (leader_current_loc[1] - follower_fixed_loc[1]) / (leader_current_loc[0] - follower_fixed_loc[0])
        dist = math.sqrt(pow(leader_current_loc[0] - follower_fixed_loc[0], 2) + pow(leader_current_loc[1] - follower_fixed_loc[1], 2))
        vector_x_cos = (leader_current_loc[0] - follower_fixed_loc[0]) / dist
        vector_x_sin = (leader_current_loc[1] - follower_fixed_loc[1]) / dist
        obstacle_nums = 0
        for x in range(0, dist, 1):
            local_map_pixel_col = x * vector_x_cos + follower_fixed_loc[0]
            local_map_pixel_row = x * vector_x_sin + follower_fixed_loc[1]
            local_map_pixel_data = follower_current_map[local_map_pixel_row][local_map_pixel_col]
            if local_map_pixel_data == 0:
                obstacle_nums += 1
                if obstacle_nums == 2:
                    return False
        return True


            


        
