# GroundingDINO ROS2 Runtime Experiments - 2026-05-27

## Runtime Query Switching

Experiment:

- Bag: `/home/boyang/safeai/converted_rosbags/outdoor_640x480_15hz`
- Replay rate: `1.0x`
- Inference resize: `image_size=480`, `max_size=640`
- Output directory: `/home/boyang/safeai/GroundingDINO/outputs/runtime_query_switch/outdoor_query_switch`
- Query schedule:
  - `0s`: `stop sign, garbage bin`
  - `14s`: `fire hydrant, construction barrel`
  - `28s`: `vehicle, stop sign`

Artifacts:

- Annotated video: `/home/boyang/safeai/GroundingDINO/outputs/runtime_query_switch/outdoor_query_switch/outdoor_query_switch_annotated.mp4`
- Query events: `/home/boyang/safeai/GroundingDINO/outputs/runtime_query_switch/outdoor_query_switch/outdoor_query_switch_query_events.jsonl`
- Structured detections: `/home/boyang/safeai/GroundingDINO/outputs/runtime_query_switch/outdoor_query_switch/outdoor_query_switch_structured_detections.jsonl`
- Runtime profile: `/home/boyang/safeai/GroundingDINO/outputs/runtime_query_switch/outdoor_query_switch/outdoor_query_switch_profile_summary.json`

Observed query changes from the node log:

- Initial: `['stop sign', 'garbage bin']`
- Switched: `['fire hydrant', 'construction barrel']`
- Switched: `['vehicle', 'stop sign']`

Output:

- Input frames observed by profiler: `599`
- Detection outputs: `593`
- Output/input ratio: `0.990`
- Output Hz: `15.05`
- Latency mean / p50 / p95: `42.6 ms / 42.4 ms / 55.7 ms`

Detection counts grouped by active query:

- `stop sign, garbage bin`: `170` frames, `593` detections
- `fire hydrant, construction barrel`: `210` frames, `635` detections
- `vehicle, stop sign`: `213` frames, `380` detections

Conclusion: runtime query switching works through the ROS2 `input_query` topic while the bag is playing.

## Runtime Profiling

Common setup:

- Bag: `/home/boyang/safeai/converted_rosbags/outdoor_640x480_15hz`
- Replay rate: `1.0x`
- Prompt: `fire hydrant, vehicle, garbage bin, construction barrel, stop sign`
- Output image disabled for the profiling-only runs
- Hardware: local RTX 4090 workstation, not Jetson Orin Nano

### Default GroundingDINO Resize

- Output directory: `/home/boyang/safeai/GroundingDINO/outputs/runtime_profile/outdoor_profile_img800_1x`
- `image_size=800`, `max_size=1333`
- Input frames: `599`
- Detection outputs: `582`
- Output/input ratio: `0.972`
- Output Hz: `14.74`
- Latency mean / p50 / p95: `89.5 ms / 85.6 ms / 125.5 ms`
- GPU memory max: `3800 MB`
- GPU utilization mean / max: `65.8% / 89.0%`

### Native 640x480-Oriented Resize

- Output directory: `/home/boyang/safeai/GroundingDINO/outputs/runtime_profile/outdoor_profile_img480_1x`
- `image_size=480`, `max_size=640`
- Input frames: `599`
- Detection outputs: `594`
- Output/input ratio: `0.992`
- Output Hz: `15.04`
- Latency mean / p50 / p95: `42.7 ms / 41.8 ms / 54.7 ms`
- GPU memory max: `3202 MB`
- GPU utilization mean / max: `29.9% / 53.0%`

Conclusion: for our 640x480 bags, setting `image_size=480` roughly halves latency and saves about `600 MB` of GPU memory on the local workstation, while still keeping the node at the source 15Hz rate.
