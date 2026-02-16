import numpy as np

from shanzi.model import ShanziModel
from shanzi.trajectory import (
    jitter_path,
    remix_sentence,
    resample_by_arclength,
    sentence_path,
    smooth_path,
)


def _toy_model() -> ShanziModel:
    tokens = ["你", "好", "世", "界"]
    vectors = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    graph = {token: [] for token in tokens}
    return ShanziModel(tokens=tokens, vectors=vectors, component_graph=graph)


def test_path_smoothing_and_resampling_shapes():
    model = _toy_model()
    path = sentence_path("你好世界", model)
    smooth = smooth_path(path, samples_per_segment=4)
    sampled = resample_by_arclength(smooth, n_samples=7)
    jittered = jitter_path(sampled, jitter_std=0.01, seed=0)

    assert path.shape == (4, 4)
    assert smooth.shape[0] > path.shape[0]
    assert sampled.shape == (7, 4)
    assert jittered.shape == sampled.shape


def test_remix_sentence_emits_known_characters():
    model = _toy_model()
    output = remix_sentence(
        "你好世界",
        model,
        samples_per_segment=4,
        output_length=6,
        jitter_std=0.0,
        seed=0,
        hanzi_only=True,
    )
    assert len(output) == 6
    assert set(output).issubset({"你", "好", "世", "界"})
