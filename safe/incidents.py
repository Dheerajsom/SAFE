"""Correlation and notification policy, independent of statistical detectors."""

from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json
import logging

logger = logging.getLogger(__name__)
SEVERITY = {"info": 0, "warning": 1, "critical": 2}


@dataclass
class HealthEvent:
    id: str
    sensor: str
    metric: str
    category: str
    severity: str
    confidence: float  # evidence-strength heuristic, never a calibrated probability
    started_at: float
    last_seen_at: float
    status: str = "open"
    evidence: dict = field(default_factory=dict)
    engine_version: str = "3.0.0"
    configuration: dict = field(default_factory=dict)
    family: str = "change"
    detection_count: int = 0
    notification_count: int = 0
    notified_at: float | None = None
    recovery_started_at: float | None = None
    recovery_count: int = 0
    closed_at: float | None = None

    def to_dict(self):
        return deepcopy(asdict(self))


class IncidentManager:
    def __init__(self, history_limit=1000, notification_interval_seconds=1800,
                 on_event=None, on_notification=None):
        if isinstance(history_limit, bool) or not isinstance(history_limit, int) or history_limit < 0:
            raise ValueError("history_limit must be a nonnegative integer")
        if (not isinstance(notification_interval_seconds, (int, float))
                or not 0 <= notification_interval_seconds < float("inf")):
            raise ValueError("notification_interval_seconds must be finite and nonnegative")
        if any(f is not None and not callable(f) for f in (on_event, on_notification)):
            raise ValueError("event handlers must be callable")
        self.history = deque(maxlen=history_limit)
        self.active = {}
        self.last_notification = {}
        self.sequence = 0
        self.total_opened = 0
        self.total_notifications = 0
        self.notification_interval_seconds = notification_interval_seconds
        self.on_event = on_event
        self.on_notification = on_notification

    def _publish(self, event, action):
        snapshot = event.to_dict()
        level = logging.DEBUG if action == "updated" else logging.INFO
        if logger.isEnabledFor(level):
            logger.log(level, json.dumps({"action": action, "event": snapshot}, allow_nan=False, sort_keys=True))
        if self.on_event:
            try:
                self.on_event(action, snapshot)
            except Exception:
                logger.exception("Health event handler failed")

    def _notify(self, event, timestamp):
        key = (event.sensor, event.metric)
        previous = self.last_notification.get(key, -float("inf"))
        if (event.notification_count or SEVERITY[event.severity] < 1
                or timestamp - previous < self.notification_interval_seconds):
            return
        event.notification_count = 1
        event.notified_at = timestamp
        self.last_notification[key] = timestamp
        self.total_notifications += 1
        if self.on_notification:
            try:
                self.on_notification(event.to_dict())
            except Exception:
                # Persist the attempt. Delivery guarantees belong to an external outbox.
                logger.exception("Health notification handler failed; attempt is not retried")

    def observe(self, sensor, metric, category, family, timestamp, severity, confidence,
                evidence, configuration):
        key = (sensor, metric, family)
        event = self.active.get(key)
        action = "updated"
        if event is None:
            self.sequence += 1
            identity = json.dumps([sensor, metric, family, timestamp, self.sequence])
            event = HealthEvent(hashlib.sha256(identity.encode()).hexdigest()[:24], sensor,
                                metric, category, severity, confidence, timestamp, timestamp,
                                configuration=deepcopy(configuration), family=family)
            self.active[key] = event
            self.total_opened += 1
            action = "opened"
        elif SEVERITY[severity] > SEVERITY[event.severity]:
            action = "escalated"
        if SEVERITY[severity] >= SEVERITY[event.severity]:
            event.category = category
            event.severity = severity
        event.confidence = max(event.confidence, confidence)
        event.last_seen_at = timestamp
        event.status = "open"
        event.recovery_started_at = None
        event.recovery_count = 0
        event.detection_count += 1
        # Fixed detector vocabulary bounds evidence growth even for long incidents.
        event.evidence[category] = {"observed_at": timestamp, **deepcopy(evidence)}
        self._notify(event, timestamp)
        self._publish(event, action)
        return event

    def recover(self, sensor, metric, timestamp, seen_families, duration, readings,
                eligible_families=None):
        for key, event in list(self.active.items()):
            if key[:2] != (sensor, metric) or event.family in seen_families:
                continue
            if eligible_families is not None and event.family not in eligible_families:
                continue
            if event.recovery_started_at is None:
                event.recovery_started_at = timestamp
                event.status = "recovering"
                self._publish(event, "recovering")
            event.recovery_count += 1
            if (timestamp - event.recovery_started_at >= duration
                    and event.recovery_count >= readings):
                event.status = "closed"
                event.closed_at = timestamp
                self.history.append(event)
                del self.active[key]
                self._publish(event, "closed")

    def events(self):
        return sorted([*self.history, *self.active.values()], key=lambda e: (e.started_at, e.id))

    def to_dict(self):
        return {"history_limit": self.history.maxlen,
                "notification_interval_seconds": self.notification_interval_seconds,
                "active": [e.to_dict() for e in self.active.values()],
                "history": [e.to_dict() for e in self.history],
                "last_notification": [[*k, v] for k, v in self.last_notification.items()],
                "sequence": self.sequence, "total_opened": self.total_opened,
                "total_notifications": self.total_notifications}

    @classmethod
    def from_dict(cls, data, **callbacks):
        obj = cls(data["history_limit"], data["notification_interval_seconds"], **callbacks)
        obj.history.extend(HealthEvent(**e) for e in data["history"])
        for item in data["active"]:
            event = HealthEvent(**item)
            obj.active[(event.sensor, event.metric, event.family)] = event
        obj.last_notification = {(s, m): t for s, m, t in data["last_notification"]}
        for key in ("sequence", "total_opened", "total_notifications"):
            setattr(obj, key, data[key])
        return obj
