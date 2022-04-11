import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import cv2

from push_primitive import PushAndAvoidTarget, PushObstacle, PushTarget
from clt_core.util.cv_tools import PinholeCameraIntrinsics, PointCloud, Feature
from clt_core.util.pybullet import get_camera_pose
from clt_core.core import MDP


def empty_push(obs, next_obs, eps=0.005):
    """
    Checks if the objects have been moved
    """

    for prev_obj in obs['full_state']['objects']:
        if prev_obj.name in ['table', 'plane']:
            continue

        for obj in next_obs['full_state']['objects']:
            if prev_obj.body_id == obj.body_id:
                if np.linalg.norm(prev_obj.pos - obj.pos) > eps:
                    return False
    return True


def get_distances_from_target(obs):
    objects = obs['full_state']['objects']

    # Get target pose from full state
    target = next(x for x in objects if x.name == 'target')
    target_pose = np.eye(4)
    target_pose[0:3, 0:3] = target.quat.rotation_matrix()
    target_pose[0:3, 3] = target.pos

    # Compute the distances of the obstacles from the target
    distances_from_target = []
    for obj in objects:
        if obj.name in ['target', 'table', 'plane']:
            continue

        # Transform the objects w.r.t. target (reduce variability)
        obj_pose = np.eye(4)
        obj_pose[0:3, 0:3] = obj.quat.rotation_matrix()
        obj_pose[0:3, 3] = obj.pos

        # distance = get_distance_of_two_bbox(target_pose, target.size, obj_pose, obj.size)
        points = p.getClosestPoints(target.body_id, obj.body_id, distance=10)
        distance = np.linalg.norm(np.array(points[0][5]) - np.array(points[0][6]))

        # points = p.getClosestPoints(target.body_id, obj.body_id, distance=10)
        # distance = np.linalg.norm(np.array(points[0][5]) - np.array(points[0][6]))

        distances_from_target.append(distance)
    return np.array(distances_from_target)


class DiscreteMDP(MDP):
    def __init__(self, params):
        super(DiscreteMDP, self).__init__('icra', params)
        # Load env params
        self.pinhole_camera_intrinsics = PinholeCameraIntrinsics.from_params(params['env']['camera']['intrinsics'])
        camera_pos, camera_quat = get_camera_pose(np.array(params['env']['camera']['pos']),
                                                  np.array(params['env']['camera']['target_pos']),
                                                  np.array(params['env']['camera']['up_vector']))
        self.camera_pose = np.eye(4)
        self.camera_pose[0:3, 0:3] = camera_quat.rotation_matrix()
        self.camera_pose[0:3, 3] = camera_pos

        self.surface_size = params['env']['workspace']['size']
        self.max_bbox = params['env']['scene_generation']['target']['max_bounding_box']

        # Load mdp params
        self.singulation_distance = params['mdp']['singulation_distance']
        self.nr_discrete_actions = params['mdp']['nr_discrete_actions']
        self.nr_primitives = params['mdp']['nr_primitives']
        self.push_distance = params['mdp']['push_distance']

        # Compute heightmap rotations
        self.rotations = int(self.nr_discrete_actions / self.nr_primitives)

    @staticmethod
    def get_heightmap(point_cloud, shape=(100, 100), grid_step=0.0035, rotations=0):
        width = shape[0]
        height = shape[1]

        height_grid = np.zeros((height, width), dtype=np.float32)

        for i in range(point_cloud.shape[0]):
            x = point_cloud[i][0]
            y = point_cloud[i][1]
            z = point_cloud[i][2]

            idx_x = int(np.floor(x / grid_step)) + int(width / 2)
            idx_y = int(np.floor(y / grid_step)) + int(height / 2)

            if 0 < idx_x < width - 1 and 0 < idx_y < height - 1:
                if height_grid[idx_y][idx_x] < z:
                    height_grid[idx_y][idx_x] = z

        if rotations > 0:
            step_angle = 360.0 / rotations
            center = (width / 2, height / 2)
            heightmaps = []
            for i in range(rotations):
                angle = i * step_angle
                rot = cv2.getRotationMatrix2D(center, angle, scale=1)
                heightmaps.append(cv2.warpAffine(height_grid, rot, (height, width)))
            return heightmaps
        else:
            return height_grid

    def get_points_above_table(self, obs):
        rgb, depth, seg = obs['rgb'], obs['depth'], obs['seg']
        objects = obs['full_state']['objects']

        # Get target pose from full state
        target = next(x for x in objects if x.name == 'target')
        target_pose = np.eye(4)
        target_pose[0:3, 0:3] = target.quat.rotation_matrix()
        target_pose[0:3, 3] = target.pos

        # Create scene point cloud
        point_cloud = PointCloud.from_depth(depth, self.pinhole_camera_intrinsics)

        # Transform point cloud w.r.t. target frame
        point_cloud.transform(np.matmul(np.linalg.inv(target_pose), self.camera_pose))

        # Keep points only above the table
        z = np.asarray(point_cloud.points)[:, 2]
        ids = np.where(z > -target.size[2])
        above_pts = point_cloud.select_by_index(ids[0].tolist())

        return np.asarray(above_pts.points)

    def state_representation(self, obs, plot=False):
        target = next(x for x in obs['full_state']['objects'] if x.name == 'target')

        # Generate heightmaps
        heightmaps = self.get_heightmap(self.get_points_above_table(obs), rotations=self.rotations)

        # Compute distances of the target from the table limits
        table = next(x for x in obs['full_state']['objects'] if x.name == 'table')
        surface_size = [table.size[0] / 2.0, table.size[1] / 2.0]  # Todo: change surface size

        distances = [surface_size[0] - target.pos[0], surface_size[0] + target.pos[0],
                     surface_size[1] - target.pos[1], surface_size[1] + target.pos[1]]
        distances = [x / (2 * surface_size[0]) for x in distances]

        bbox = [target.size[0] / self.max_bbox[0], target.size[1] / self.max_bbox[1]]

        rotation_angles = np.arange(0.0, 360.0, 360.0 / self.rotations)

        features = []
        for i in range(self.rotations):

            if plot:
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(heightmaps[i])
                ax[1].imshow(Feature(heightmaps[i]).crop(32, 32).pooling(kernel=[4, 4], stride=4, mode='AVG').array())
                plt.show()

            # Crop and avg pool each heightmap to generate a 16x16 map
            z = Feature(heightmaps[i]).crop(32, 32).pooling(kernel=[4, 4], stride=4, mode='AVG').flatten()
            z /= 2 * self.max_bbox[2]

            # The final feature is a concatenation of the flattened heightmap, distances from the table limits,
            # target's bounding bos and the corresponding rotation angle
            features.append(np.concatenate((z, np.asarray(distances), np.asarray(bbox),
                                            np.asarray([rotation_angles[i] / 360.0]))))

        return np.asarray(features)

    def reward(self, obs, next_obs, action):
        if next_obs['collision'] or empty_push(obs, next_obs):
            return -15.0

        # Fall off the table
        target = next(x for x in next_obs['full_state']['objects'] if x.name == 'target')
        if target.pos[2] < 0.0:
            return -15.0

        # Break the cluster extra penalty
        extra_penalty = 0
        if action > 15:
            extra_penalty = -4.0

        # Singulation
        distances_from_target = get_distances_from_target(next_obs)
        if all(dist > self.singulation_distance for dist in distances_from_target):
            return 10.0 + extra_penalty

        return -1.0 + extra_penalty

    def terminal(self, obs, next_obs):
        """
        Parameters
        ----------
        obs
        next_obs

        Returns
        -------
        terminal : int
            0: not a terminal state
            2: singulation
            3: fallen-off the table
        """
        # In case the target is singulated or falls of the table the episode is singulated
        # ToDo: end the episode for maximum number of pushes

        if next_obs['collision']:
            return -1

        if empty_push(obs, next_obs):
            return -2

        distances_from_target = get_distances_from_target(next_obs)
        target = next(x for x in next_obs['full_state']['objects'] if x.name == 'target')
        if target.pos[2] < 0:
            return 3
        elif all(dist > self.singulation_distance for dist in distances_from_target):
            return 2
        else:
            return 0

    def action(self, obs, action):
        target = next(x for x in obs['full_state']['objects'] if x.name == 'target')

        # Compute the discrete angle theta
        theta = action * 2 * np.pi / (self.nr_discrete_actions / self.nr_primitives)

        # Choose the action
        if action > 2 * int(self.nr_discrete_actions / self.nr_primitives) - 1:
            # Break the cluster: 16-23
            push = PushTarget()
            push(theta=theta, push_distance=self.push_distance, distance=self.params['mdp']['init_distance'])
        elif action > 1 * int(self.nr_discrete_actions / self.nr_primitives) - 1:
            # Push obstacle: 8-15
            push = PushObstacle()
            push(theta=theta, push_distance=self.push_distance, target_size_z=target.size[2])
        else:
            # Push target: 0-7
            push = PushAndAvoidTarget(finger_size=obs['full_state']['finger'])
            push(theta=theta, push_distance=self.push_distance, distance=0.0,
                 convex_hull=target.convex_hull(oriented=False))

        push.transform(target.pos, target.quat)

        return push.p1.copy(), push.p2.copy()

    def init_state_is_valid(self, obs):
        distances_from_target = get_distances_from_target(obs)
        if all(dist > self.singulation_distance for dist in distances_from_target):
            return False
        return True
