"""One bounded, low-priority process for throttled whole-map highway work."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import os
import time

import numpy as np

from mapping.slam_map import FREE
from navigation.highway import HighwayBuildResult, build_highway_graph


# One preparation for one exact whole-map free mask, never a set of old graphs.
_preparation_cache = {}


def _initialize_worker():
    """Limit competing native threads and lower this optional worker's priority."""
    import cv2
    cv2.setNumThreads(1)
    try:
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
        else:
            os.nice(5)
    except (OSError, AttributeError):
        pass


def _build(slam, settings):
    return build_highway_graph(slam, maximum_connector_expansions=4096,
                               preparation_cache=_preparation_cache, **settings)


class HighwayWorker:
    """Coalesce uploads into one latest input and allow only one running job."""

    MINIMUM_INTERVAL = 2.0

    def __init__(self, settings, *, executor=None, now=time.monotonic):
        self.settings = settings
        self.executor = executor or ProcessPoolExecutor(max_workers=1, initializer=_initialize_worker)
        self.now = now
        self.latest = None
        self.latest_free = self.published_free = None
        self.latest_geometry = self.published_geometry = None
        self.future = None
        self.job_slam = None
        self.job_free = None
        self.job_version = self.job_geometry = None
        self.job_retry = False
        self.retry = None
        self.last_submission = -float("inf")
        self.closed = False

    def request(self, slam, current):
        self.latest = slam
        self.latest_geometry = (slam.origin, slam.full_shape or slam.occupancy.shape)
        self.latest_free = ((slam.occupancy == FREE) &
                            (slam.confidence >= self.settings["confidence_threshold"]))
        if current is not None and self._matches_publication():
            graph = replace(current, version=slam.version)
            return HighwayBuildResult("cached", graph, 0, graph.area_count,
                                      graph.edge_count, graph.known_free_cells,
                                      source_version=slam.version, build_kind="cached")
        self._submit()
        return HighwayBuildResult("pending" if not self.closed else "unavailable", None, 0,
                                  known_free_cells=int(np.count_nonzero(self.latest_free)),
                                  source_version=slam.version)

    def _matches_publication(self):
        return (self.published_free is not None and self.latest_geometry == self.published_geometry
                and np.array_equal(self.latest_free, self.published_free))

    def _safe_subset(self, free, geometry):
        return (geometry == self.latest_geometry and free.shape == self.latest_free.shape
                and not np.any(free & ~self.latest_free))

    def _submit(self):
        now = self.now()
        if self.closed or self.future is not None or self.latest is None or now - self.last_submission < self.MINIMUM_INTERVAL:
            return
        if self._matches_publication():
            self.retry = None
            return
        self.job_slam = self.latest
        self.job_free = self.latest_free
        self.job_geometry = self.latest_geometry
        self.job_retry = False
        if self.retry is not None:
            slam, free, geometry = self.retry
            self.retry = None
            if self._safe_subset(free, geometry):
                self.job_slam, self.job_free, self.job_geometry = slam, free, geometry
                self.job_retry = True
        self.job_version = self.job_slam.version
        self.last_submission = now
        try:
            self.future = self.executor.submit(_build, self.job_slam, self.settings)
        except (RuntimeError, OSError):
            self.closed = True

    def poll(self):
        result = None
        if self.future is not None and self.future.done():
            try:
                result = self.future.result()
            except Exception:
                # Advice is optional; a worker failure must not interrupt reports.
                result = HighwayBuildResult("unavailable", None, 0)
                self.closed = True
            self.future = None
            result = replace(result, source_version=self.job_version, build_kind="full")
            if (result.status == "budget_exhausted" and not self.job_retry
                    and self._safe_subset(self.job_free, self.job_geometry)):
                # Finish one exact input before newer additions replace its
                # cached preparation. A wall correction cancels this retry.
                self.retry = (self.job_slam, self.job_free, self.job_geometry)
            if result.snapshot is not None:
                same_geometry = (self.job_geometry == self.latest_geometry
                                 and self.job_free.shape == self.latest_free.shape)
                unchanged = same_geometry and np.array_equal(self.job_free, self.latest_free)
                # Additions do not invalidate this coherent older snapshot.
                # Publishing it lets advice progress during continuous uploads;
                # the newest additions remain queued for the next full rebuild.
                safe_subset = self._safe_subset(self.job_free, self.job_geometry)
                if safe_subset:
                    graph = replace(result.snapshot, version=self.latest.version) if unchanged else result.snapshot
                    result = replace(result, snapshot=graph)
                    self.published_free = self.latest_free if unchanged else self.job_free
                    self.published_geometry = self.job_geometry
                else:
                    # Stale results cannot roll back a corrected wall or a
                    # newer publication.
                    result = replace(result, status="superseded", snapshot=None)
        self._submit()
        return result

    def shutdown(self):
        self.closed = True
        if self.future is not None:
            self.future.cancel()
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.future = self.latest = self.latest_free = self.job_free = self.job_slam = self.retry = None
