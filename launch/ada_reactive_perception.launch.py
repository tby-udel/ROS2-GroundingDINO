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
            "image_size",
            default_value="800",
            description="GroundingDINO resize short-edge target before inference",
        ),
        DeclareLaunchArgument(
            "max_size",
            default_value="1333",
            description="GroundingDINO maximum long-edge size before inference",
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
            "frame_stride",
            default_value="1",
            description="Process one image every N incoming frames",
        ),
        DeclareLaunchArgument(
            "max_detections",
            default_value="100",
            description="Maximum detections retained after thresholding; 0 keeps all",
        ),
        DeclareLaunchArgument(
            "empty_cache_every_n_frames",
            default_value="0",
            description="Call torch.cuda.empty_cache every N processed frames; 0 disables",
        ),
        DeclareLaunchArgument(
            "torch_num_threads",
            default_value="0",
            description="Set PyTorch CPU worker threads when >0",
        ),
        DeclareLaunchArgument(
            "disable_model_checkpointing",
            default_value="true",
            description="Disable inference-time checkpoint wrappers where present",
        ),
        DeclareLaunchArgument(
            "input_image_topic",
            default_value="/camera/camera/color/image_raw",
            description="Input image topic",
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
            default_value="true",
            description="Publish compatibility outputs on legacy topics",
        ),
        DeclareLaunchArgument(
            "publish_legacy_image",
            default_value="true",
            description="Publish annotated compatibility image on legacy_image_topic",
        ),
        DeclareLaunchArgument(
            "legacy_detection_topic",
            default_value="/yolo/detections",
            description="Compatibility detection topic",
        ),
        DeclareLaunchArgument(
            "legacy_image_topic",
            default_value="/yolo/inference_image",
            description="Compatibility image topic",
        ),
    ]

    groundingdino_node = Node(
        package="ros2_groundingdino",
        executable="groundingdino_py",
        remappings=[("input_image", LaunchConfiguration("input_image_topic"))],
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
                "image_size": ParameterValue(LaunchConfiguration("image_size"), value_type=int),
                "max_size": ParameterValue(LaunchConfiguration("max_size"), value_type=int),
                "device": LaunchConfiguration("device"),
                "precision": LaunchConfiguration("precision"),
                "frame_stride": ParameterValue(LaunchConfiguration("frame_stride"), value_type=int),
                "max_detections": ParameterValue(LaunchConfiguration("max_detections"), value_type=int),
                "empty_cache_every_n_frames": ParameterValue(
                    LaunchConfiguration("empty_cache_every_n_frames"),
                    value_type=int,
                ),
                "torch_num_threads": ParameterValue(LaunchConfiguration("torch_num_threads"), value_type=int),
                "disable_model_checkpointing": ParameterValue(
                    LaunchConfiguration("disable_model_checkpointing"),
                    value_type=bool,
                ),
                "initial_query": LaunchConfiguration("initial_query"),
                "publish_output_image": ParameterValue(LaunchConfiguration("publish_output_image"), value_type=bool),
                "publish_legacy_outputs": ParameterValue(LaunchConfiguration("publish_legacy_outputs"), value_type=bool),
                "publish_legacy_image": ParameterValue(LaunchConfiguration("publish_legacy_image"), value_type=bool),
                "legacy_detection_topic": LaunchConfiguration("legacy_detection_topic"),
                "legacy_image_topic": LaunchConfiguration("legacy_image_topic"),
            }
        ],
    )

    return LaunchDescription(launch_args + [groundingdino_node])
