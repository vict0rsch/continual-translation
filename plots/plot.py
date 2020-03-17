import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path

if __name__ == "__main__":

    plt.imsave(
        "plots/concat_1.png",
        np.concatenate(
            [
                np.concatenate([a[:, :256, :], a[:, 768:, :]], axis=1)
                for i, f in enumerate(
                    sorted(str(d) for d in Path("./plots/base64").glob("*1.png"))
                )
                for a in [np.array(Image.open(f))[-256 if i > 0 else -512 :, :1279, :]]
            ],
            axis=0,
        ),
    )

    plt.imsave(
        "plots/concat_2.png",
        np.concatenate(
            [
                np.concatenate([a[:, 256:768, :], a[:, 1024:, :]], axis=1)
                for i, f in enumerate(
                    sorted(str(d) for d in Path("./plots/base64").glob("*2.png"))
                )
                for a in [np.array(Image.open(f))[-256 if i > 0 else -512 :, :1279, :]]
            ],
            axis=0,
        ),
    )

    plt.imsave(
        "plots/concats.png",
        np.concatenate(
            [
                np.array(Image.open("plots/concat_1.png"))[:1022, :, :],
                np.array(Image.open("plots/concat_2.png"))[:1022, :, :],
            ],
            axis=1,
        ),
    )
