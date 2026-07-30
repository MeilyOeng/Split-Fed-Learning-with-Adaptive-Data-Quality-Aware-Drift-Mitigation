"""Isolated River detector state with absolute-coordinate safeguards."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

from .contracts import DetectorEvent, DetectorKey, DetectorName, DetectorParameters


def _build_detector(name: DetectorName, parameters: DetectorParameters):
    try:
        from river.drift import ADWIN, PageHinkley
    except ImportError as exc:
        raise RuntimeError("Page-Hinkley and ADWIN require the 'river' package") from exc
    if name is DetectorName.PAGE_HINKLEY:
        return PageHinkley(
            min_instances=parameters.ph_min_instances,
            delta=parameters.ph_delta,
            threshold=parameters.ph_threshold,
            alpha=parameters.ph_alpha,
        )
    return ADWIN(
        delta=parameters.adwin_delta,
        clock=parameters.adwin_clock,
        max_buckets=parameters.adwin_max_buckets,
        min_window_length=parameters.adwin_min_window_length,
        grace_period=parameters.adwin_grace_period,
    )


@dataclass
class _DetectorState:
    detector: object
    last_coordinate: Optional[int] = None
    events: deque[DetectorEvent] = field(default_factory=deque)
    alarms: list[int] = field(default_factory=list)
    event_count: int = 0


class DetectorRegistry:
    """Own detector instances independently for every complete DetectorKey."""

    def __init__(self, parameters: Optional[DetectorParameters] = None, *, max_buffered_events: int = 10000, event_sink: Optional[Callable[[DetectorEvent], None]] = None) -> None:
        self.parameters = parameters or DetectorParameters()
        if max_buffered_events <= 0:
            raise ValueError("max_buffered_events must be > 0")
        self.max_buffered_events = int(max_buffered_events)
        self.event_sink = event_sink
        self._states: dict[DetectorKey, _DetectorState] = {}

    def _state(self, key: DetectorKey) -> _DetectorState:
        if key not in self._states:
            self._states[key] = _DetectorState(_build_detector(key.detector_name, self.parameters), events=deque(maxlen=self.max_buffered_events))
        return self._states[key]

    def update(
        self,
        key: DetectorKey,
        *,
        observation_index: int,
        value: float,
        timestamp: Optional[str] = None,
        regime_active: Optional[bool] = None,
        transform_applied: Optional[bool] = None,
    ) -> Optional[DetectorEvent]:
        if observation_index < 0:
            raise ValueError("observation_index must be >= 0")
        if not math.isfinite(float(value)):
            raise ValueError("detector value must be finite")
        state = self._state(key)
        if state.last_coordinate is not None:
            if observation_index < state.last_coordinate:
                raise ValueError(
                    f"decreasing detector coordinate for {key}: {observation_index} < {state.last_coordinate}"
                )
            if observation_index == state.last_coordinate:
                return None
        state.detector.update(float(value))
        alarm = bool(state.detector.drift_detected)
        event = DetectorEvent(
            run_id=key.run_id,
            cluster_id=key.cluster_id,
            client_id=key.client_id,
            mode=key.mode,
            stream_name=key.stream_name,
            detector_name=key.detector_name.value,
            event_index=state.event_count,
            observation_index=observation_index,
            timestamp=timestamp,
            detector_value=float(value),
            alarm=alarm,
            regime_active=regime_active,
            transform_applied=transform_applied,
            detector_min_instances=(self.parameters.ph_min_instances if key.detector_name is DetectorName.PAGE_HINKLEY else None),
        )
        state.events.append(event)
        state.event_count += 1
        if self.event_sink is not None:
            self.event_sink(event)
        if alarm:
            state.alarms.append(observation_index)
        state.last_coordinate = observation_index
        return event

    def events(self, key: DetectorKey) -> tuple[DetectorEvent, ...]:
        state = self._states.get(key)
        return tuple(state.events) if state is not None else ()

    def alarms(self, key: DetectorKey) -> tuple[int, ...]:
        state = self._states.get(key)
        return tuple(state.alarms) if state is not None else ()

    def drain_events(self, key: DetectorKey) -> tuple[DetectorEvent, ...]:
        state = self._states.get(key)
        if state is None:
            return ()
        events = tuple(state.events)
        state.events.clear()
        return events

    def reset(self, key: Optional[DetectorKey] = None) -> None:
        if key is None:
            self._states.clear()
        else:
            self._states.pop(key, None)

    def keys(self) -> tuple[DetectorKey, ...]:
        return tuple(self._states)
