"""The voice endpoint and its lease (SPEC §8): the armed device is the only
client allowed to capture and play audio; the lease survives connection
blips and expires after 60 s unreachable, freeing the toggle everywhere."""

from phd_helper.endpoint import VoiceEndpoint


def test_first_client_to_arm_holds_the_voice_endpoint():
    endpoint = VoiceEndpoint()

    assert endpoint.arm("desktop", now=0) is True
    assert endpoint.arm("phone", now=0) is False
    assert endpoint.is_view_only("desktop", now=0) is False
    assert endpoint.is_view_only("phone", now=0) is True


def test_the_lease_survives_a_connection_blip():
    endpoint = VoiceEndpoint()
    endpoint.heartbeat("desktop", now=0)
    endpoint.arm("desktop", now=0)
    endpoint.heartbeat("phone", now=0)

    # desktop unreachable 10 s — dead by the heartbeat rule, inside the blip
    assert endpoint.endpoint(now=10) == "desktop"
    assert endpoint.arm("phone", now=10) is False


def test_the_lease_expires_after_sixty_seconds_unreachable(tmp_path):
    endpoint = VoiceEndpoint()
    endpoint.heartbeat("desktop", now=0)
    endpoint.arm("desktop", now=0)
    endpoint.heartbeat("phone", now=0)

    assert endpoint.endpoint(now=61) is None
    assert endpoint.arm("phone", now=61) is True
    assert endpoint.endpoint(now=61) == "phone"


def test_a_returning_departed_device_is_view_only():
    endpoint = VoiceEndpoint()
    endpoint.heartbeat("desktop", now=0)
    endpoint.arm("desktop", now=0)
    endpoint.heartbeat("phone", now=0)
    endpoint.arm("phone", now=61)  # phone takes the freed channel

    endpoint.heartbeat("desktop", now=70)  # desktop returns

    assert endpoint.is_view_only("desktop", now=70) is True
    assert endpoint.endpoint(now=70) == "phone"


def test_a_server_restart_never_rearms_the_microphone():
    endpoint = VoiceEndpoint()
    endpoint.heartbeat("desktop", now=0)
    endpoint.arm("desktop", now=0)

    restarted = VoiceEndpoint()

    assert restarted.endpoint(now=1) is None
    assert restarted.is_view_only("desktop", now=1) is True


def test_a_client_is_declared_dead_after_two_missed_heartbeats():
    endpoint = VoiceEndpoint(ping_interval=2.0)
    endpoint.heartbeat("desktop", now=0)

    assert endpoint.is_connected("desktop", now=3.9) is True
    assert endpoint.is_connected("desktop", now=4.1) is False
