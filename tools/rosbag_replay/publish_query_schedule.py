#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class QuerySchedulePublisher(Node):
    def __init__(self, topic, events, repeat_count, repeat_period):
        super().__init__("groundingdino_query_schedule_publisher")
        self.publisher = self.create_publisher(String, topic, 10)
        self.events = events
        self.repeat_count = repeat_count
        self.repeat_period = repeat_period

    def publish_query(self, query):
        msg = String()
        msg.data = query
        for _ in range(self.repeat_count):
            self.publisher.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(self.repeat_period)
        self.get_logger().info(f"Published query: {query}")


def parse_schedule(schedule):
    events = []
    for item in schedule.split(";"):
        item = item.strip()
        if not item:
            continue
        if "|" not in item:
            raise ValueError(f"Schedule item must be '<delay_seconds>|<query>': {item}")
        delay_text, query = item.split("|", 1)
        events.append((float(delay_text.strip()), query.strip()))
    return sorted(events, key=lambda event: event[0])


def main():
    parser = argparse.ArgumentParser(description="Publish timed std_msgs/String query updates for GroundingDINO.")
    parser.add_argument("--topic", default="/input_query")
    parser.add_argument("--schedule", required=True, help="Semicolon separated '<delay_seconds>|<query>' entries.")
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--repeat-count", type=int, default=3)
    parser.add_argument("--repeat-period", type=float, default=0.15)
    args = parser.parse_args()

    events = parse_schedule(args.schedule)
    log_path = Path(args.log_path).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = QuerySchedulePublisher(args.topic, events, args.repeat_count, args.repeat_period)
    started_at = time.monotonic()

    with log_path.open("w", encoding="utf-8") as log_file:
        for delay, query in events:
            while time.monotonic() - started_at < delay:
                rclpy.spin_once(node, timeout_sec=0.05)
                time.sleep(0.05)
            record = {
                "scheduled_delay_seconds": delay,
                "published_unix_time": time.time(),
                "query": query,
            }
            node.publish_query(query)
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_file.flush()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
