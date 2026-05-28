from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            "groundingdino_dir",
            default_value="/home/boyang/safeai/GroundingDINO",
            description="Path to the local GroundingDINO checkout",
        ),
        DeclareLaunchArgument(
            "config",
            default_value="/home/boyang/safeai/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            description="GroundingDINO model config",
        ),
        DeclareLaunchArgument(
            "checkpoint",
            default_value="/home/boyang/safeai/GroundingDINO/weights/groundingdino_swint_ogc.pth",
            description="GroundingDINO checkpoint",
        ),
        DeclareLaunchArgument(
            "thresholds",
            default_value="0.3",
            description="NanoOWL-compatible detection threshold",
        ),
        DeclareLaunchArgument(
            "text_threshold",
            default_value="0.25",
            description="GroundingDINO phrase grounding threshold",
        ),
        DeclareLaunchArgument(
            "device",
            default_value="cuda",
            description="Inference device",
        ),
        DeclareLaunchArgument(
            "precision",
            default_value="fp32",
            description="Inference precision: fp32 or fp16",
        ),
        DeclareLaunchArgument(
            "initial_query",
            default_value="a person, a box",
            description="Initial comma-separated open-vocabulary query",
        ),
        DeclareLaunchArgument(
            "publish_output_image",
            default_value="false",
            description="Publish annotated image to output_image",
        ),
        DeclareLaunchArgument(
            "publish_legacy_outputs",
            default_value="false",
            description="Publish compatibility outputs on legacy topics",
        ),
    ]

    groundingdino_node = Node(
        package="ros2_groundingdino",
        executable="groundingdino_py",
        parameters=[
            {
                "groundingdino_dir": LaunchConfiguration("groundingdino_dir"),
                "config": LaunchConfiguration("config"),
                "checkpoint": LaunchConfiguration("checkpoint"),
                "model": LaunchConfiguration("checkpoint"),
                "image_encoder_engine": "",
                "thresholds": ParameterValue(LaunchConfiguration("thresholds"), value_type=float),
                "box_threshold": ParameterValue(LaunchConfiguration("thresholds"), value_type=float),
                "text_threshold": ParameterValue(LaunchConfiguration("text_threshold"), value_type=float),
                "device": LaunchConfiguration("device"),
                "precision": LaunchConfiguration("precision"),
                "initial_query": LaunchConfiguration("initial_query"),
                "publish_output_image": ParameterValue(LaunchConfiguration("publish_output_image"), value_type=bool),
                "publish_legacy_outputs": ParameterValue(LaunchConfiguration("publish_legacy_outputs"), value_type=bool),
            }
        ],
    )

    return LaunchDescription(launch_args + [groundingdino_node])
