import torch


class kcs_util:
    def __init__(self, joints, bones, partitions, num_joints=None):
        self.num_joints = num_joints
        self.joints = joints
        self.bones = bones
        self.partitions = partitions
        self.C = None

    def extend(self, x):
        return x

    def compute_features(self):
        return (self.num_joints - 1) ** 2

    def c(self, joint_i, joint_j):
        ls = [0 for i in range(self.num_joints)]
        ls[self.joints[joint_i]] = 1
        ls[self.joints[joint_j]] = -1
        return ls

    def bone_vectors(self, ext_input):
        if self.C is None:
            self.C = torch.tensor([self.c(bone[0], bone[1]) for bone in self.bones]).transpose(1, 0)\
                     .type(torch.FloatTensor)

        C = self.C.repeat([ext_input.size(0), 1, 1])
        ext_input = ext_input.permute(0, 2, 1).type(torch.FloatTensor)
        B = torch.matmul(ext_input, C)
        B = B.permute(0, 2, 1)
        return B

    def center(self, inputs_3d):
        index = self.joints['Hips']
        return inputs_3d - inputs_3d[:, index: index + 1, :]

    def kcs_layer(self, bv, region):
        index = self.partitions[region]
        mask = torch.zeros_like(bv)
        mask[:, index, :] = 1
        bv = bv * mask
        kcs = torch.matmul(bv, bv.permute(0, 2, 1))
        return kcs


class kcs_util17(kcs_util):
    def __init__(self, joints, bones, partitions):
        super().__init__(joints, bones, partitions, num_joints=17)


class kcs_util13(kcs_util):
    def __init__(self, joints, bones, partitions):
        super().__init__(joints, bones, partitions, num_joints=16)

    def extend(self, x):
        X = torch.empty((x.size(0), self.num_joints, 3))
        X[:, 0:x.size()[1], :] = x
        X[:, self.joints['Hips'], :] = (x[:, self.joints['Lhip'], :] + x[:, self.joints['Rhip'], :]) / 2
        X[:, self.joints['Neck'], :] = (x[:, self.joints['Lshoulder'], :] + x[:, self.joints['Rshoulder'], :]) / 2
        X[:, self.joints['Spine'], :] = (X[:, self.joints['Neck'], :] + X[:, self.joints['Hips'], :]) / 2
        return X


def KCS_util(num_joints=17):
    if num_joints == 13:
        joints = {'Nose': 0,
                  'Lshoulder': 1,
                  'Rshoulder': 2,
                  'Lelbow': 3,
                  "Relbow": 4,
                  'Lwrist': 5,
                  'Rwrist': 6,
                  'Lhip': 7,
                  'Rhip': 8,
                  'Lknee': 9,
                  'Rknee': 10,
                  'Lankle': 11,
                  'Rankle': 12,
                  # These joints are computed by us
                  'Hips': 13,
                  'Neck': 14,
                  'Spine': 15
                  }

        bones = [('Hips', 'Lhip'), ('Lhip', 'Lknee'), ('Lknee', 'Lankle'),  # Left Leg Done
                 ('Hips', 'Rhip'), ('Rhip', 'Rknee'), ('Rknee', 'Rankle'),  # Right Leg Done
                 ('Hips', 'Spine'), ('Spine', 'Neck'), ('Neck', 'Nose'),  # Spine done
                 ('Nose', 'Lshoulder'), ('Lshoulder', 'Lelbow'), ('Lelbow', 'Lwrist'),  # Left shoulder done
                 ('Nose', 'Rshoulder'), ('Rshoulder', 'Relbow'), ('Relbow', 'Rwrist')  # Right shoulder done
                 ]

        partitions = {"ll": [0, 1, 2, 6],
                      "rl": [3, 4, 5, 6],
                      "torso": [0, 3, 6, 7, 8, 9, 12],
                      "lh": [7, 9, 10, 11],
                      "rh": [7, 12, 13, 14]}

        return kcs_util13(joints, bones, partitions)

    elif num_joints == 17:
        joints = {'Hips': 0,
                  'RightUpLeg': 1,
                  'RightLeg': 2,
                  'RightFoot': 3,
                  'LeftUpLeg': 4,
                  'LeftLeg': 5,
                  'LeftFoot': 6,
                  'Spine': 7,
                  'Neck': 8,
                  'Head': 9,
                  'Site-head': 10,
                  'LeftArm': 11,
                  'LeftForeArm': 12,
                  'LeftHand': 13,
                  'RightArm': 14,
                  'RightForeArm': 15,
                  'RightHand': 16
                  }
        bones = [('Hips', 'LeftUpLeg'), ('LeftUpLeg', 'LeftLeg'), ('LeftLeg', 'LeftFoot'),  # Left Leg is done
                 ('Hips', 'RightUpLeg'), ('RightUpLeg', 'RightLeg'), ('RightLeg', 'RightFoot'),  # Right Leg is done
                 ('Hips', 'Spine'), ('Spine', 'Neck'), ('Neck', 'Head'), ('Head', 'Site-head'),  # Spine is done
                 ('Neck', 'LeftArm'), ('LeftArm', 'LeftForeArm'), ('LeftForeArm', 'LeftHand'),  # Left Arm is done
                 ('Neck', 'RightArm'), ('RightArm', 'RightForeArm'), ('RightForeArm', 'RightHand')]  # Right Arm is done

        partitions = {"ll": [0, 1, 2, 6],
                      "rl": [3, 4, 5, 6],
                      "torso": [0, 3, 6, 7, 8, 9, 10, 13],
                      "lh": [7, 10, 11, 12],
                      "rh": [7, 13, 14, 15]}

        return kcs_util17(joints, bones, partitions)

    else:
        raise NotImplemented
