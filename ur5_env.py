'''
 @Author: Prince Wang
 @Date: 2024-02-22
 @Last Modified by:   Prince Wang
 @Last Modified time: 2023-10-24 23:04:04
'''
import os
from ur5_config import setup_sisbot
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pybullet as p
import pybullet_data
import math
import time
from pybullet_utils import bullet_client
from scipy.spatial.transform import Rotation as R
from loguru import logger
import random


class UR5_Env(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, gui=True,gripper_type='85'):
        super(UR5_Env).__init__()
        self.step_num = 0

        # 设置最小的关节变化量
        low_action = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
        high_action = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        low_action = np.array([-0.1, -0.1, -0.1, -0.1, -0.1, -0.1])
        high_action = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        # 定义了一个Box类型的动作空间
        self.action_space = spaces.Box(low=low_action, high=high_action, dtype=np.float32)

        low = (-6)*np.ones((1, 12), dtype=np.float32)
        high = 6*np.ones((1, 12), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        if gui == False:
            self.p = bullet_client.BulletClient(connection_mode=p.DIRECT)
        # DIRECT模式创建一个新的物理引擎并直接与之通信。
        # GUI将创建一个带有图形GUI前端的物理引擎并与之通信。
        else:
            self.p = bullet_client.BulletClient(connection_mode=p.GUI)
        self.p.setTimeStep(1 / 240)
        # print(self.p)
        self.p.setGravity(0, 0, -9.81)
        self.p.resetSimulation()
        # 设置搜索路径以加载模型
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # boxId = self.p.loadURDF("plane.urdf")
        # 创建机械臂，并启用碰撞检测
        # self.ur5 = self.p.loadURDF("urdf/ur5_robotiq_85.urdf", useFixedBase=True,
        #                            basePosition=[0, 0, 0],
        #                            baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2]),
        #                            flags=p.URDF_USE_SELF_COLLISION)

        self.ur5 = p.loadURDF("./urdf/ur5_robotiq_%s.urdf" % gripper_type,
                                  [0, 0, 0],  # StartPosition
                                  p.getQuaternionFromEuler([0, 0, 0]),  # StartOrientation
                                  useFixedBase=True,
                                  flags=p.URDF_USE_SELF_COLLISION)
        self.joints, self.controlGripper, self.controlJoints, self.mimicParentName = \
            setup_sisbot(p, self.ur5, gripper_type)
        self.eefID = 7  # ee_link
        # Add force sensors
        p.enableJointForceTorqueSensor(self.ur5, self.joints['left_inner_finger_pad_joint'].id)
        p.enableJointForceTorqueSensor(self.ur5, self.joints['right_inner_finger_pad_joint'].id)

        p.changeDynamics(self.ur5, self.joints['left_inner_finger_pad_joint'].id, lateralFriction=0.5)
        p.changeDynamics(self.ur5, self.joints['right_inner_finger_pad_joint'].id, lateralFriction=0.5)
        # 创建桌子
        collisionTableId = self.p.createCollisionShape(shapeType=p.GEOM_BOX,
                                                       halfExtents=[0.9, 0.6, 0.5])  # 半长宽高
        # self.table = self.p.createMultiBody(baseMass=0,  # 质量
        #                                     baseCollisionShapeIndex=collisionTableId,
        #                                     basePosition=[0.0, -0.5, 0.8]
        #
        #                                     )  # 基础位置（确保立方体底部与地面平齐）
        self.table = p.loadURDF("./urdf/objects/table.urdf",
                                  [0.0, -0.5, 0.8],  # base position
                                  p.getQuaternionFromEuler([0, 0, 0]),  # base orientation
                                  useFixedBase=True)
        self.UR5StandID = p.loadURDF("./urdf/objects/ur5_stand.urdf",
                                     [-0.7, -0.36, 0.0],
                                     p.getQuaternionFromEuler([0, 0, 0]),
                                     useFixedBase=True)
        # 创建目标
        collisionTargetId = self.p.createCollisionShape(shapeType=p.GEOM_CYLINDER,
                                                        radius=0.02, height=0.05)
        pos=0.2
        self.target = self.p.createMultiBody(baseMass=0,  # 质量
                                             baseCollisionShapeIndex=collisionTargetId,
                                             basePosition = [0, -0.5, 1.15] )

        # 创建目标杯子的台子
        collisionTargetId = self.p.createCollisionShape(shapeType=p.GEOM_CYLINDER,
                                                        radius=0.03, height=0.3)
        self.targettable = self.p.createMultiBody(baseMass=0,  # 质量
                                                  baseCollisionShapeIndex=collisionTargetId,
                                                  basePosition = [0, -0.5, 1.15] )

        # collisionTargetId1 = self.p.createCollisionShape(shapeType=p.GEOM_SPHERE,
        #                                     radius=0.01)
        # self.show = self.p.createMultiBody(baseMass=0,  # 质量
        #                     baseCollisionShapeIndex=collisionTargetId1,
        #                     basePosition=[0.5, 0.5, 2])

    def step(self, action):
        info = {}
        total_reward = 0
        success = False
        # Execute one time step within the environment
        # 初始化关节角度列表
        joint_angles = []

        # 获取每个关节的状态
        self.move_ee(joint_angles,action)
        # ur5_joint_angles = np.array(joint_angles) + (np.array(action[0:6]) / 180 * np.pi)
        # gripper = np.array([0, 0])
        # anglenow = np.hstack([ur5_joint_angles, gripper])
        # # anglenow[5] = -92/180*np.pi
        # p.setJointMotorControlArray(self.ur5, [1, 2, 3, 4, 5, 6, 8, 9], p.POSITION_CONTROL, targetPositions=anglenow)

        '''
            机械臂执行运动
        '''
        for _ in range(20):
            self.p.stepSimulation()
            # time.sleep(1./240.)
        left_index = self.joints['left_inner_finger_pad_joint'].id
        right_index = self.joints['right_inner_finger_pad_joint'].id
        # 计算奖励
        # 1.计算碰撞奖励
        # 若机械臂成功抓取目标，那么任务成功
        # 若机械臂发生其他碰撞（桌子或其他关节碰撞目标），那么任务失败
        gripper_joint_indices = [left_index, right_index]
        target_contact_points = p.getContactPoints(bodyA=self.ur5, bodyB=self.target)
        table_contact_points = p.getContactPoints(bodyA=self.ur5, bodyB=self.table)
        self_contact_points = p.getContactPoints(bodyA=self.ur5, bodyB=self.ur5)
        self_targettable_contact_points = p.getContactPoints(bodyA=self.ur5, bodyB=self.targettable)

        # 定义碰撞变量
        gripper_contact = False
        other_contact = False
        target_contact = False
        other_contact1 = False

        # 机械臂关节碰撞目标
        for contact_point in target_contact_points:
            link_index = contact_point[3]  # getContactPoints返回的第四个信息为bodyA的碰撞关节数
            if link_index ==left_index   or link_index == right_index :
                gripper_contact = True
                logger.info("机械臂夹爪接触目标!")
            # 检查是否有非夹爪关节接触目标
            if link_index not in gripper_joint_indices and gripper_contact == False:
                target_contact = True
                logger.info("机械臂接触目标！")

        # 碰撞桌子
        for contact_point in table_contact_points:
            link_index = contact_point[3]
            if link_index == 0 or link_index == 1:
                other_contact1 = False
            else:
                other_contact = True
                logger.info("碰撞桌子！")

        # # 碰撞自身
        # for contact_point in self_contact_points:
        #     link_indexA = contact_point[3]
        #     link_indexB = contact_point[4]
        #     if (link_indexA == 5 and link_indexB == 7) or (link_indexA == 8 and link_indexB == 9) or (
        #             link_indexA == 2 and link_indexB == 7) or (
        #             link_indexA == 4 and link_indexB == 7) or link_indexB == 7:
        #         other_contact1 = False
        #     else:
        #         other_contact = True
        #         logger.info("碰撞自身! ")

        # 碰撞目标杯子的台子
        for contact_point in self_targettable_contact_points:
            other_contact = True
            logger.info("碰撞目标杯子的台子! ")

        # 2.判断机械臂与夹爪的距离
        # Gripper_posx = (p.getLinkState(self.fr5, 8)[0][0]+p.getLinkState(self.fr5, 9)[0][0])/2
        # Gripper_posy = (p.getLinkState(self.fr5, 8)[0][1]+p.getLinkState(self.fr5, 9)[0][1])/2
        # Gripper_posz = (p.getLinkState(self.fr5, 8)[0][2]+p.getLinkState(self.fr5, 9)[0][2])/2
        Gripper_posx = p.getLinkState(self.ur5, 7)[0][0]
        Gripper_posy = p.getLinkState(self.ur5, 7)[0][1]
        Gripper_posz = p.getLinkState(self.ur5, 7)[0][2]
        relative_position = np.array([0, 0, 0.15])#偏移量
        # 固定夹爪相对于机械臂末端的相对位置转换
        rotation = R.from_quat(p.getLinkState(self.ur5, 9)[1])
        rotated_relative_position = rotation.apply(relative_position)
        gripper_centre_pos = [Gripper_posx, Gripper_posy, Gripper_posz] + rotated_relative_position
        distance = math.sqrt((gripper_centre_pos[0] - self.goalx) ** 2 + (gripper_centre_pos[1] - self.goaly) ** 2 + (
                    gripper_centre_pos[2] - self.goalz) ** 2)
        # logger.debug("distance:%s"%str(distance))

        # self.p.resetBasePositionAndOrientation(selfz'z.show,gripper_centre_pos, [0, 0, 0, 1])
        # self.p.stepSimulation()

        # 判断成功或失败
        if distance < 0.015:
            success = True
        else:
            success = False
            # total_reward = total_reward + (0.3 - distance)

        # 计算距离奖励
        if self.step_num == 0:
            distance_reward = 0
        else:
            if distance <= 0.41:
                distance_reward = 1000 * (self.distance_last - distance)
            elif distance > 0.41:
                distance_reward = -2 * pow(math.e, 4.3 * distance)
        #     logger.debug("相对距离：%f"%(self.distance_last-distance))
        # logger.debug("距离奖励:%f"%distance_reward)

        # 保存上一次的距离
        self.distance_last = distance

        # 3.姿态奖励
        # 计算夹爪的朝向
        gripper_orientation = p.getLinkState(self.ur5, 9)[1]
        gripper_orientation = R.from_quat(gripper_orientation)
        gripper_orientation = gripper_orientation.as_euler('xyz', degrees=True)
        # print("夹爪旋转角度： ",gripper_orientation)
        # 计算夹爪的姿态奖励
        pose_reward = -(pow(gripper_orientation[0] + 90, 2) + pow(gripper_orientation[1], 2) * 5 + pow(
            gripper_orientation[2] + 90, 2) * 5) * 0.005
        # logger.debug("姿态奖励：%f"%pose_reward)

        total_reward = total_reward + pose_reward + distance_reward

        # all:判断成功或失败
        # 并计算成功或失败的奖励

        # 如果两个夹爪都接触目标，那么任务成功
        if success == True and self.step_num <= 100:
            total_reward = total_reward + 1000
            self.terminated = True
            self.success = True
            info['is_success'] = True
            info['step_num'] = self.step_num
            logger.info("成功抓取！！！！！！！！！！执行步数：%s  距离目标:%s" % (self.step_num, distance))
            # self.truncated = True

        # 碰撞桌子，或者碰撞自身，或者碰撞台子
        elif other_contact:
            total_reward = total_reward - 100
            self.terminated = True
            info['is_success'] = False
            info['step_num'] = self.step_num
            logger.info("失败！执行步数：%s    距离目标:%s" % (self.step_num, distance))
            # self.truncated = True

        # 机械臂关节接触目标
        elif target_contact:
            total_reward = total_reward - 100
            self.terminated = True
            info['is_success'] = False
            info['step_num'] = self.step_num
            logger.info("失败！执行步数：%s    距离目标:%s" % (self.step_num, distance))

        # 机械臂夹爪接触目标
        elif gripper_contact:
            total_reward = total_reward - 80
            self.terminated = True
            info['is_success'] = False
            info['step_num'] = self.step_num
            logger.info("失败！执行步数：%s    距离目标:%s" % (self.step_num, distance))

        # 机械臂执行步数过多
        elif self.step_num > 100:
            total_reward = total_reward - 100
            self.terminated = True
            info['is_success'] = False
            info['step_num'] = self.step_num
            logger.info("执行步数超时！")
            logger.info("失败！执行步数：%s    距离目标:%s" % (self.step_num, distance))

        self.truncated = False
        self.reward = total_reward
        info['reward'] = self.reward
        # print(self.reward)

        # observation计算
        gripper_centre_pos[0] = self.add_noise(gripper_centre_pos[0], range=0.005, gaussian=True)
        gripper_centre_pos[1] = self.add_noise(gripper_centre_pos[1], range=0.005, gaussian=True)
        gripper_centre_pos[2] = self.add_noise(gripper_centre_pos[2], range=0.005, gaussian=True)
        obs_gripper_centre_pos = np.array([(gripper_centre_pos[0] + 0.922) / 1.844,
                                           (gripper_centre_pos[1] + 0.922) / 1.844,
                                           (gripper_centre_pos[2] + 0.5) / 1], dtype=np.float32)

        obs_gripper_orientation = (np.array([gripper_orientation[0], gripper_orientation[1], gripper_orientation[2]],
                                            dtype=np.float32) + 180) / 360



        obs_target_position = np.array([(self.target_position[0] + 0.2) / 0.4,
                                        (self.target_position[1] + 0.7) / 0.4,
                                        (self.target_position[2] - 1.05) / 0.25], dtype=np.float32)

        joint_angles = [0, 0, 0, 0, 0, 0]
        for i in [1, 2, 3, 4, 5, 6]:
            joint_info = p.getJointState(self.ur5, i)
            joint_angles[i - 1] = joint_info[0] * 180 / np.pi  # 第一个元素是当前关节角度
            joint_angles[i - 1] = self.add_noise(joint_angles[i - 1], range=0.5, gaussian=True)

        obs_joint_angles = ((np.array(joint_angles, dtype=np.float32) / 180) + 1) / 2

        # print("joint_angles",str(joint_angles))
        # print("gripper_centre_pos",str(gripper_centre_pos))

        self.observation = np.hstack((obs_joint_angles, obs_gripper_centre_pos, obs_target_position),
                                     dtype=np.float32).flatten()

        self.observation = self.observation.flatten()
        self.observation = self.observation.reshape(1, 12)
        # self.observation = np.hstack((np.array(joint_angles,dtype=np.float32),target_delta_position[0]),dtype=np.float32)
        self.step_num += 1

        return self.observation, self.reward, self.terminated, self.truncated, info

    def reset(self, seed=None, options=None):
        self.step_num = 0
        self.reward = 0
        # self.stepcount = 0
        self.terminated = False
        # 重新设置机械臂的位置
        # neutral_angle = [-49.45849125928217, -57.601209583849, -138.394013961943, -164.0052115563118,
        #                  -49.45849125928217, -90, 0, 0]
        # # neutral_angle =[ 0,0,0,0,0,0,0,0]
        # neutral_angle = [x * math.pi / 180 for x in neutral_angle]
        # p.setJointMotorControlArray(self.ur5, [1, 2, 3, 4, 5, 6, 8, 9], p.POSITION_CONTROL,
        #                             targetPositions=0)
        # 初始位置
        user_parameters = (-1.5690622952052096, -1.5446774605904932, 1.343946009733127, -1.3708613585093699,
                           -1.5707970583733368, 0.0009377758247187636, 0.085)
        for i, name in enumerate(self.controlJoints):
            if i == 6:
                self.controlGripper(controlMode=p.POSITION_CONTROL, targetPosition=user_parameters[i])
                break
            joint = self.joints[name]
                # control robot joints
            p.setJointMotorControl2(self.ur5, joint.id, p.POSITION_CONTROL,
                                        targetPosition=user_parameters[i], force=joint.maxForce,
                                        maxVelocity=joint.maxVelocity)
                # self.p.stepSimulation()
        # neutral_angle =[ -49.45849125928217, -57.601209583849, -138.394013961943, -164.0052115563118,-49.45849125928217,0,0]
        # # neutral_angle =[ 0,0,0,0,0,0,0,0]
        # neutral_angle = [x * math.pi / 180 for x in neutral_angle]
        # p.setJointMotorControlArray(self.fr5,[1,2,3,4,5,8,9],p.POSITION_CONTROL,targetPositions=neutral_angle)
        # p.setJointMotorControl2(self.fr5,6,p.POSITION_CONTROL,targetPosition=-50*math.pi/180,force = 100)

        # # 重新设置目标位置
        # self.goaly = np.random.uniform(-0.7, -0.3, 1)
        # self.goalx = np.random.uniform(-0.2, 0.2, 1)
        # self.goalz = np.random.uniform(1.05, 1.3, 1)
        self.goalx= 0
        self.goaly= -0.5
        self.goalz = 1.15
        self.target_position = [self.goalx, self.goaly, self.goalz]
        self.targettable_position = [self.goalx, self.goaly, self.goalz - 0.175]
        self.p.resetBasePositionAndOrientation(self.targettable, self.targettable_position, [0, 0, 0, 1])
        self.p.resetBasePositionAndOrientation(self.target, self.target_position, [0, 0, 0, 1])

        for i in range(100):
            self.p.stepSimulation()
            # time.sleep(10./240.)

        # 计算observation
        # Gripper_posx = (p.getLinkState(self.fr5, 8)[0][0]+p.getLinkState(self.fr5, 9)[0][0])/2
        # Gripper_posy = (p.getLinkState(self.fr5, 8)[0][1]+p.getLinkState(self.fr5, 9)[0][1])/2
        # Gripper_posz = (p.getLinkState(self.fr5, 8)[0][2]+p.getLinkState(self.fr5, 9)[0][2])/2
        Gripper_posx = p.getLinkState(self.ur5, 7)[0][0]
        Gripper_posy = p.getLinkState(self.ur5, 7)[0][1]
        Gripper_posz = p.getLinkState(self.ur5, 7)[0][2]
        relative_position = np.array([0, 0, 0.15])

        # 固定夹爪相对于机械臂末端的相对位置转换
        rotation = R.from_quat(p.getLinkState(self.ur5, 9)[1])
        rotated_relative_position = rotation.apply(relative_position)
        # print([Gripper_posx, Gripper_posy,Gripper_posz])
        gripper_centre_pos = [Gripper_posx, Gripper_posy, Gripper_posz] + rotated_relative_position

        # self.p.resetBasePositionAndOrientation(self.show,gripper_centre_pos, [0, 0, 0, 1])
        # self.p.stepSimulation()

        joint_angles = [0, 0, 0, 0, 0, 0]
        for i in [1, 2, 3, 4, 5, 6]:
            joint_info = p.getJointState(self.ur5, i)
            joint_angles[i - 1] = joint_info[0] * 180 / np.pi  # 第一个元素是当前关节角度
        # print("joint_angles",str(joint_angles))
        # print("gripper_centre_pos",str(gripper_centre_pos))

        # 计算夹爪的朝向
        gripper_orientation = p.getLinkState(self.ur5, 9)[1]
        gripper_orientation = R.from_quat(gripper_orientation)
        gripper_orientation = gripper_orientation.as_euler('xyz', degrees=True)

        # 计算obs
        obs_joint_angles = ((np.array(joint_angles, dtype=np.float32) / 180) + 1) / 2

        obs_gripper_centre_pos = np.array([(gripper_centre_pos[0] + 0.922) / 1.844,
                                           (gripper_centre_pos[1] + 0.922) / 1.844,
                                           (gripper_centre_pos[2] + 0.5) / 1], dtype=np.float32)

        obs_gripper_orientation = (np.array([gripper_orientation[0], gripper_orientation[1], gripper_orientation[2]],
                                            dtype=np.float32) + 180) / 360

        obs_target_position = np.array([(self.target_position[0] +0.2) / 0.4,
                                        (self.target_position[1] + 0.7) / 0.4,
                                        (self.target_position[2] - 1.05) / 0.25], dtype=np.float32)

        self.observation = np.hstack((obs_joint_angles, obs_gripper_centre_pos, obs_target_position),
                                     dtype=np.float32).flatten()

        self.observation = self.observation.flatten()
        self.observation = self.observation.reshape(1, 12)
        # self.observation = np.hstack((np.array(joint_angles,dtype=np.float32),target_delta_position[0]),dtype=np.float32)

        info = {}
        info['is_success'] = False
        info['reward'] = 0
        info['step_num'] = 0
        return self.observation, info

    def render(self):
        p.resetDebugVisualizerCamera(
            cameraDistance=1.7, cameraYaw=0, cameraPitch=-20, cameraTargetPosition=[0.45, 0, 0.8])

    def close(self):
        self.p.disconnect()

    def add_noise(self, angle, range, gaussian=False):
        if gaussian:
            angle += np.clip(np.random.normal(0, 1) * range, -1, 1)
        else:
            angle += random.uniform(-5, 5)
        return angle

    def move_ee(self, joint_angles,action ,max_step=500, custom_velocity=5,
               ):

        # print(self.controlJoints)
        # print(self.controlJoints[:-1])
        # for _ in range(max_step):
            # apply IK

        for i, name in enumerate(self.controlJoints[:-1]):  # Filter out the gripper

            joint = self.joints[name]
            joint_info = p.getJointState(self.ur5, self.joints[name].id)
            joint_angle = joint_info[0]  # 第一个元素是当前关节角度
            joint_angles.append(joint_angle)
                # ur5_joint_angles = np.array(joint_angles) + (np.array(action[0:6]) / 180 * np.pi)
                # gripper = np.array([0])
                # anglenow = np.hstack([ur5_joint_angles, gripper])
            #anglenow=joint_angles.append(0)
               # print(anglenow)
                # control robot end-effector
            p.setJointMotorControl2(self.ur5, joint.id, p.POSITION_CONTROL,
                                        targetPosition=joint_angle+action[i], force=joint.maxForce,
                                        maxVelocity=joint.maxVelocity if custom_velocity is None else custom_velocity * (i+1))

            #self.p.stepSimulation()




if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    Env = UR5_Env(gui=True)
    Env.reset()
    check_env(Env, warn=True)
    # for i in range(100):
    #         p.stepSimulation()
    #         time.sleep(1./240.)
    Env.render()
    print("test going")
    time.sleep(10)
    # observation, reward, terminated, truncated, info = Env.step([0,0,0,0,0,20])
    # print(reward)

