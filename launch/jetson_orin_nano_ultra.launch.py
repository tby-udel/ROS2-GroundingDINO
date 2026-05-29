from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            "groundingdino_dir",
            default_value="/home/ada2/GroundingDINO",
            description="Path to the local GroundingDINO checkout on the Jetson",
        ),
        DeclareLaunchArgument(
            "config",
            default_value="/home/ada2/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            description="GroundingDINO model config",
        ),
        DeclareLaunchArgument(
            "checkpoint",
            default_value="/home/ada2/GroundingDINO/weights/groundingdino_swint_ogc.pth",
            description="GroundingDINO checkpoint",
        ),
        DeclareLaunchArgument(
            "input_image_topic",
            default_value="/camera/camera/color/image_raw",
            description="Input image topic",
        ),
        DeclareLaunchArgument(
            "initial_query",
            default_value="stop sign, garbage bin",
            description="Initial comma-separated open-vocabulary query",
        ),
        DeclareLaunchArgument(
            "thresholds",
            default_value="0.35",
            description="Higher threshold reduces output load and false positives",
        ),
        DeclareLaunchArgument(
            "text_threshold",
            default_value="0.25",
            description="GroundingDINO phrase grounding threshold",
        ),
        DeclareLaunchArgument(
            "image_size",
            default_value="224",
            description="Aggressive short-edge resize for Jetson Orin Nano 8GB",
        ),
        DeclareLaunchArgument(
            "max_size",
            default_value="320",
            description="Aggressive long-edge cap for Jetson Orin Nano 8GB",
        ),
        DeclareLaunchArgument(
            "frame_stride",
            default_value="3",
            description="Process one image every N incoming frames",
        ),
        DeclareLaunchArgument(
            "max_detections",
            default_value="20",
            description="Maximum detections retained after thresholding",
        ),
        DeclareLaunchArgument(
            "empty_cache_every_n_frames",
            default_value="8",
            description="Release cached CUDA blocks periodically on memory-constrained Jetsons",
        ),
        DeclareLaunchArgument(
            "torch_num_threads",
            default_value="2",
            description="Limit CPU thread pressure on Jetson",
        ),
        DeclareLaunchArgument(
            "device",
            default_value="cuda",
            description="Inference device",
        ),
        DeclareLaunchArgument(
            "precision",
            default_value="fp32",
            description="Keep FP32 by default because PyTorch FP16 failed on Jetson in testing",
        ),
        DeclareLaunchArgument(
            "publish_output_image",
            default_value="false",
            description="Disable annotated output image for minimum memory and CPU load",
        ),
        DeclareLaunchArgument(
            "publish_legacy_outputs",
            default_value="true",
            description="Keep ADAONE-compatible detection string output",
        ),
        DeclareLaunchArgument(
            "publish_legacy_image",
            default_value="false",
            description="Disable legacy annotated image for strongest compression",
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
                "frame_stride": ParameterValue(LaunchConfiguration("frame_stride"), value_type=int),
                "max_detections": ParameterValue(LaunchConfiguration("max_detections"), value_type=int),
                "empty_cache_every_n_frames": ParameterValue(
                    LaunchConfiguration("empty_cache_every_n_frames"),
                    value_type=int,
                ),
                "torch_num_threads": ParameterValue(LaunchConfiguration("torch_num_threads"), value_type=int),
                "disable_model_checkpointing": True,
                "device": LaunchConfiguration("device"),
                "precision": LaunchConfiguration("precision"),
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
