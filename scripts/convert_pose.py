#!/usr/bin/env python3
import json
import sys
import copy

DEFAULT_GIMBAL_PITCH = 0.0
DEFAULT_SHOOT = True

def convert(input_file, output_file, gimbal_pitch=DEFAULT_GIMBAL_PITCH, shoot=DEFAULT_SHOOT):
    with open(input_file, 'r') as f:
        poses = json.load(f)

    result = []
    for pose_obj in poses:
        new_pose = copy.deepcopy(pose_obj)
        for key in new_pose:
            new_pose[key].setdefault('gimbalPitch', gimbal_pitch)
            new_pose[key].setdefault('shoot', shoot)
        result.append(new_pose)

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Converted {len(result)} poses -> {output_file}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.json> <output.json> [gimbalPitch] [shoot]")
        print(f"  gimbalPitch: default {DEFAULT_GIMBAL_PITCH}")
        print(f"  shoot:       default {DEFAULT_SHOOT}")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    pitch = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_GIMBAL_PITCH
    shoot = sys.argv[4].lower() in ('true', '1', 'yes') if len(sys.argv) > 4 else DEFAULT_SHOOT

    convert(input_file, output_file, pitch, shoot)
