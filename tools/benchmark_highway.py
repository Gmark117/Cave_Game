"""Compare tiled and branching highways on identical frozen SLAM inputs.

Run with synthetic fixtures, or pass an NPZ containing occupancy, confidence,
and optionally origin/full_shape. No cave generation or mission execution runs.
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mapping.slam_map import FREE, OCCUPIED, SlamSnapshot
from navigation.highway import build_highway_graph, build_tiled_highway_graph


def benchmark_updates(name, slam, reach, *, background=False):
    """Measure throttled full publication and one localized wall correction."""
    from dataclasses import replace
    from navigation.highway import HighwayService
    from navigation.highway_worker import _build
    service = HighwayService(confidence_threshold=.6, macro_cell_size=32,
                             maximum_build_ms=250, maximum_query_ms=50,
                             maximum_connector_expansions=4096,
                             maximum_access_distance_sensor_ranges=1, sensor_range=reach)
    settings = dict(confidence_threshold=.6, macro_cell_size=32,
                    maximum_build_ms=250, maximum_connector_expansions=4096,
                    maximum_access_distance=reach)
    if background:
        service.start()

    def publish(label, source):
        attempts = []
        started = time.perf_counter()
        request_ms = None
        if background:
            service.refresh(source)
            request_ms = (time.perf_counter() - started) * 1000
        graph = None
        while time.perf_counter() - started < 15:
            worker_settings = {key: value for key, value in settings.items() if key != "maximum_connector_expansions"}
            result = service.poll() if background else _build(source, worker_settings)
            if result is None:
                time.sleep(.01)
                continue
            attempts.append(dict(status=result.status, build_ms=round(result.elapsed_ms, 2)))
            if result.snapshot is not None:
                graph = result.snapshot
                break
            if result.status == "unavailable":
                break
        print(json.dumps(dict(fixture=name, stage=label, background=background,
                              request_ms=None if request_ms is None else round(request_ms, 2),
                              publication_ms=round((time.perf_counter() - started) * 1000, 2),
                              attempts=attempts, version=None if graph is None else graph.version,
                              nodes=0 if graph is None else graph.area_count,
                              edges=0 if graph is None else graph.edge_count,
                              maximum_access_px=None if graph is None else graph.measured_access_distance)))
        if graph is None:
            raise RuntimeError("full graph did not publish within 15 seconds")
        assert not graph.area_labels.flags.writeable
        return graph

    try:
        publish("initial", slam)
        free = (slam.occupancy == FREE) & (slam.confidence >= .6)
        interior = np.zeros_like(free)
        interior[1:-1, 1:-1] = (free[1:-1, 1:-1] & free[:-2, 1:-1] & free[2:, 1:-1]
                                & free[1:-1, :-2] & free[1:-1, 2:])
        points = np.argwhere(interior)
        if len(points):
            y, x = points[len(points) // 2]
            occupancy = slam.occupancy.copy()
            occupancy[y, x] = OCCUPIED
            correction = replace(slam, occupancy=occupancy, version=slam.version + 1)
            graph = publish("local_wall_correction", correction)
            assert not graph._area_at(graph._local((int(x + slam.origin[0]), int(y + slam.origin[1]))))
    finally:
        service.shutdown()


def replay_snapshot(trace_path, map_path, scan_limit):
    """Reconstruct sensor belief and deliver it only at recorded contacts.

    The caller must supply the matching cave. This is a deterministic offline
    reconstruction, not an exact capture of the original concurrent run.
    """
    from mapping.slam_map import SlamMap
    from mapping.vision_sensor import VisionSensor
    from mapping.drone_sensor import LIDAR_RANGE_RADIUS_MULTIPLIER

    cave = np.loadtxt(map_path).astype(np.uint8)
    drones, rover, sensor = [], SlamMap(*cave.shape), None
    scans = contacts = 0
    with trace_path.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            name = event["event"]
            if name == "mission_constructed":
                if cave.shape != (event["map_height"], event["map_width"]):
                    raise ValueError("trace and cave dimensions do not match")
                radius = {"SMALL": 40, "MEDIUM": 20, "LARGE": 10}[event["map_dim"]]
                sensor = VisionSensor(cave, max_range=radius * LIDAR_RANGE_RADIUS_MULTIPLIER)
                drones = [SlamMap(*cave.shape) for _ in range(event["drones"])]
            elif name == "sensor_scan":
                x, y, heading = event["pose"]
                observation = sensor.scan_cone((x, y), heading)
                drones[event["drone_id"]].update_from_observations(
                    (x, y), free_cells=observation.free_cells,
                    occupied_cells=observation.occupied_cells,
                )
                scans += 1
                if scans >= scan_limit:
                    break
            elif name == "drone_slam_exchange":
                first, second = drones[event["drone_id"]], drones[event["other_drone_id"]]
                a, b = first.snapshot(point_limit=0), second.snapshot(point_limit=0)
                if event.get("drone_slam_changed"):
                    first.merge_from(b)
                if event.get("other_slam_changed"):
                    second.merge_from(a)
            elif name in {"drone_rover_check_in", "drone_rover_departure_share",
                          "drone_rover_proximity_share"} and event.get("arrived"):
                drone = drones[event["drone_id"]]
                a, b = drone.snapshot(point_limit=0), rover.snapshot(point_limit=0)
                rover.merge_from(a)
                drone.merge_from(b)
                contacts += 1
    snapshot = rover.snapshot(point_limit=0)
    print(json.dumps(dict(reconstructed_trace=trace_path.name, scans=scans,
                          physical_rover_encounters=contacts,
                          known_free_cells=int(np.count_nonzero(snapshot.occupancy == FREE)))))
    return snapshot


def fixtures():
    wall = np.full((160, 224), FREE, np.int8)
    wall[:120, 104:120] = OCCUPIED
    ring = np.full((100, 140), OCCUPIED, np.int8)
    ring[10:90, 10:130] = FREE
    ring[30:70, 40:100] = OCCUPIED
    branches = np.full((200, 240), OCCUPIED, np.int8)
    branches[80:110, 10:230] = FREE
    branches[20:180, 60:90] = FREE
    branches[40:150, 170:200] = FREE
    for name, occupancy in (("wall", wall), ("loop", ring), ("branches", branches)):
        yield name, SlamSnapshot(occupancy, np.ones(occupancy.shape, np.float32), version=1)


def compare(name, slam, reach):
    graphs, rows = [], []
    _, components = cv2.connectedComponents((slam.occupancy == FREE).astype(np.uint8), connectivity=4)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    points = np.argwhere(components == int(np.argmax(sizes))) if sizes.max() else np.empty((0, 2), int)
    rng = np.random.default_rng(0)
    selected = rng.choice(len(points), min(24, len(points)), replace=False)
    ox, oy = slam.origin
    pairs = [((int(a[1] + ox), int(a[0] + oy)), (int(b[1] + ox), int(b[0] + oy)))
             for a, b in zip(points[selected[::2]], points[selected[1::2]])]
    if name == "wall":
        pairs.insert(0, ((20, 20), (200, 20)))
    all_routes = []
    for builder in (build_tiled_highway_graph, build_highway_graph):
        settings = dict(confidence_threshold=0.6, macro_cell_size=32,
                        maximum_build_ms=250.0, maximum_connector_expansions=4096)
        if builder is build_highway_graph:
            settings["maximum_access_distance"] = reach
        result = builder(slam, **settings)
        graph = result.snapshot
        routes = [] if graph is None else [graph.route(
            start, goal, maximum_query_ms=50.0, maximum_connector_expansions=4096,
        ) for start, goal in pairs]
        graphs.append(graph)
        all_routes.append(routes)
        rows.append(dict(builder=builder.__name__, status=result.status,
                         build_ms=round(result.elapsed_ms, 2), nodes=result.area_count,
                         edges=result.edge_count, complete_routes=sum(route.complete for route in routes),
                         maximum_access_px=None if graph is None else graph.measured_access_distance))
    matched = [(a.cost, b.cost) for a, b in zip(*all_routes) if a.complete and b.complete]
    print(json.dumps(dict(fixture=name, access_limit_px=reach, results=rows,
                          matched_routes=len(matched), tiled_distance_px=round(sum(a for a, _ in matched), 2),
                          backbone_distance_px=round(sum(b for _, b in matched), 2))))
    return graphs, all_routes


def panel(slam, graph, route):
    canvas = np.full((*slam.occupancy.shape, 3), 30, np.uint8)
    canvas[slam.occupancy == OCCUPIED] = 90
    canvas[slam.occupancy == FREE] = 240
    ox, oy = slam.origin
    def line(path, color, width):
        points = np.array([(x - ox, y - oy) for x, y in path], np.int32)
        if len(points) > 1:
            cv2.polylines(canvas, [points], False, color, width)
    if graph is not None:
        for source, edges in enumerate(graph.adjacency):
            for edge in edges:
                if source < edge.target_area:
                    line(edge.path, (160, 160, 160), 1)
    if route is not None and route.complete:
        line(route.path, (60, 60, 210), 2)
    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError("could not encode highway preview")
    return base64.b64encode(encoded).decode("ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--map", type=Path)
    parser.add_argument("--scan-limit", type=int, default=180)
    parser.add_argument("--access-distance", type=float, default=80.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--updates", action="store_true", help="benchmark throttled full publication instead of tiled comparison")
    parser.add_argument("--background", action="store_true", help="use the mission's separate highway process with --updates")
    args = parser.parse_args()
    if args.background and not args.updates:
        parser.error("--background requires --updates")
    if args.trace and not args.map:
        parser.error("--trace requires --map with the matching cave matrix")
    cases = list(fixtures())
    if args.trace:
        cases.append(("replayed-rover", replay_snapshot(args.trace, args.map, args.scan_limit)))
    if args.snapshot:
        with np.load(args.snapshot, allow_pickle=False) as data:
            cases.append(("snapshot", SlamSnapshot(
                data["occupancy"], data["confidence"], version=1,
                origin=tuple(map(int, data["origin"])) if "origin" in data else (0, 0),
                full_shape=tuple(map(int, data["full_shape"])) if "full_shape" in data else None,
            )))
    elements, top = [], 0
    for name, slam in cases:
        if args.updates:
            benchmark_updates(name, slam, args.access_distance, background=args.background)
            continue
        graphs, routes = compare(name, slam, args.access_distance)
        height, width = slam.occupancy.shape
        scaled_height = 480 * height / max(1, width)
        for index, label in enumerate(("Tiled", "Branching")):
            route = routes[index][0] if routes[index] else None
            cost = f"{route.cost:.1f}px" if route is not None and route.complete else "fallback"
            elements.append(f'<text x="{index * 500 + 10}" y="{top + 22}" font-size="18">{name}: {label} ({cost})</text>')
            elements.append(f'<image x="{index * 500 + 10}" y="{top + 32}" width="480" height="{scaled_height}" href="data:image/png;base64,{panel(slam, graphs[index], route)}"/>')
        top += scaled_height + 55
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{top}" viewBox="0 0 1000 {top}"><rect width="100%" height="100%" fill="white"/>{"".join(elements)}</svg>',
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
