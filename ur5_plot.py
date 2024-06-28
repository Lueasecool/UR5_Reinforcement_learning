import pybullet as p
import time
import pybullet_data
import math
from collections import namedtuple
from attrdict import AttrDict
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
#p.setGravity(0,0,-9.8)
planeId = p.loadURDF("plane.urdf")

robotId = p.loadURDF("urdf/ur5_robotiq_85.urdf",useFixedBase = True) #set the center of mass frame (loadURDF sets base link frame)

# 登记各个节点的信息
jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
numJoints = p.getNumJoints(robotId)
jointInfo = namedtuple("jointInfo", ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity"])
joints = AttrDict()
for i in range(numJoints):
    info = p.getJointInfo(robotId, i)
    jointID = info[0]
    jointName = info[1].decode('utf-8')
    jointType = jointTypeList[info[2]]
    jointLowerLimit = info[8]
    jointUpperLimit = info[9]
    jointMaxForce = info[10]
    jointMaxVelocity = info[11]
    singleInfo = jointInfo(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce,
                           jointMaxVelocity)
    joints[singleInfo.name] = singleInfo

print(joints)

for jointID in joints:
    print("jointName:", jointName,
          "\n jointID:",jointID)


p.disconnect()