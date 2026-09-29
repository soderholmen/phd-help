"""The voice endpoint and its lease (SPEC §8).

A pure state machine over an explicit clock: arming is the only way in, the
lease survives connection blips, and only lease expiry frees it for an
explicit handoff.
"""


class VoiceEndpoint:
    def __init__(self, ping_interval: float = 2.0, lease_timeout: float = 60.0):
        self._ping_interval = ping_interval
        self._lease_timeout = lease_timeout
        self._holder: str | None = None
        self._last_seen: dict[str, float] = {}

    def heartbeat(self, client_id: str, now: float) -> None:
        self._last_seen[client_id] = now

    def is_connected(self, client_id: str, now: float) -> bool:
        """Two missed round trips declare the connection dead."""
        return now - self._last_seen.get(client_id, now) <= 2 * self._ping_interval

    def endpoint(self, now: float) -> str | None:
        """The current holder; the lease expires only after the holder has
        been unreachable past the timeout (a blip keeps it)."""
        if self._holder is not None:
            unreachable = now - self._last_seen.get(self._holder, now)
            if unreachable > self._lease_timeout:
                self._holder = None
        return self._holder

    def arm(self, client_id: str, now: float) -> bool:
        if self.endpoint(now) is not None and self._holder != client_id:
            return False
        self._holder = client_id
        self._last_seen[client_id] = now  # arming is itself proof of liveness
        return True

    def is_view_only(self, client_id: str, now: float) -> bool:
        return self.endpoint(now) != client_id
