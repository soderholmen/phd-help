"""The Qwen3-Reranker adapter, minus the model: the GPU probe proved
CrossEncoder speaks this model natively; what's under test here is the
adapter's own contract — margins in, probabilities out, in input order.

Probabilities matter: the pinned-doc boost is a multiplier, and
multiplying a signed margin by 1.5 would push a pinned doc DOWN. The
sigmoid keeps the scale positive so boosting can only ever lift.
"""

from phd_helper.server.rerank import QwenReranker


class FakeCE:
    def __init__(self):
        self.pairs = None

    def predict(self, pairs):
        self.pairs = list(pairs)
        return [9.0, -9.0, 0.0]


def test_margins_squash_to_probabilities_in_order():
    r = QwenReranker()
    ce = FakeCE()
    r._model = ce  # bypass the lazy GPU load
    scores = r.rerank("q", ["a", "b", "c"])
    assert ce.pairs == [("q", "a"), ("q", "b"), ("q", "c")]
    assert scores[0] > 0.99 and scores[1] < 0.01
    assert scores[2] == 0.5


def test_boost_multiplier_only_ever_lifts():
    r = QwenReranker()
    r._model = FakeCE()
    scores = r.rerank("q", ["a", "b", "c"])
    assert all(s * 1.5 > s for s in scores)
