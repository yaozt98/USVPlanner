from pathlib import Path
import numpy as np
import pysocialforce as psf


if __name__ == "__main__":
    # initial states, each entry is the position, velocity and goal of a pedestrian in the form of (px, py, vx, vy, gx, gy)
    initial_state = np.array(
        [
            [0.0, 10, -0.5, -0.5, 0.0, 0.0],
            [0.5, 10, -0.5, -0.5, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.5, 1.0, 10.0],
        ]
    )

    # initiate the simulator,
    s = psf.Simulator(
        initial_state,
        delta_t = 0.2,
        config_file=Path(__file__).resolve().parent.joinpath("socialforce_example.toml"),
    )
    s.step(50)

    with psf.plot.SceneVisualizer(s, "images/exmaple3") as sv:
        sv.animate()
        # sv.plot()