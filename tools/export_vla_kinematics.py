"""Export the installed URDF TCP chain to JSON using the system ROS environment."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'wheel_arm_manipulation'))
from ament_index_python.packages import get_package_share_directory
from wheel_arm_manipulation.planning import load_model
from wheel_arm_manipulation.grasp_kinematics import GraspKinematics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args=parser.parse_args()
    model=load_model()
    fk=GraspKinematics(model)
    source=Path(get_package_share_directory('wheel_arm_description'))/'urdf/wheel_arm.urdf.xacro'
    spec={'schema_version':1,'frame':model.frame,'tip':'gripper_tcp','joint_names':list(model.joints),
          'urdf_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'chain':[{'name':n,'kind':k,'origin':o.tolist(),'axis':None if a is None else a.tolist()}
                   for n,k,o,a in fk.chain]}
    with Path(args.output).open('x') as stream:
        json.dump(spec,stream,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
