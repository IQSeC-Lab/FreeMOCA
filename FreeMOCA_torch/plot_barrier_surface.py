import os, glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def load_curves(folder="barrier_curves"):
    files = sorted(glob.glob(os.path.join(folder, "curve_*_to_*.npz")))
    curves = []
    for f in files:
        base = os.path.basename(f).replace(".npz", "")
        parts = base.split("_")
        t = int(parts[1])  # curve_t_to_t1
        data = np.load(f)
        curves.append({
            "t": t,
            "s": data["s"],
            "loss": data["loss"],
            "acc": data["acc"],
            "barrier": float(data["barrier"][0]),
        })
    curves = sorted(curves, key=lambda d: d["t"])
    return curves

def make_surface(curves, use_relative=True):
    s = curves[0]["s"]
    T = len(curves)

    X = np.arange(T)
    Y = s
    XX, YY = np.meshgrid(X, Y, indexing="xy")

    Z = np.zeros_like(YY, dtype=float)

    for j, c in enumerate(curves):
        loss = c["loss"].copy()
        if use_relative:
            baseline = 0.5 * (loss[0] + loss[-1])
            loss = loss - baseline
        Z[:, j] = loss

    return XX, YY, Z

def plot_surface(curves, title="Connected valleys (loss along paths)", use_relative=True,
                 surface_cmap="Blues_r", surface_alpha=0.55):
    XX, YY, Z = make_surface(curves, use_relative=use_relative)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # --- Surface: light color ---
    surf = ax.plot_surface(
        XX, YY, Z,
        rstride=1, cstride=1,
        linewidth=0, antialiased=True,
        alpha=surface_alpha,
        cmap=surface_cmap
    )

    ax.set_xlabel("Transition index (t→t+1)")
    ax.set_ylabel("Path position s")
    ax.set_zlabel("ΔLoss" if use_relative else "Loss")
    ax.set_title(title)

    # --- X markers: deep/dark colors ---
    # Use one strong color per transition (deep palette)
    deep_colors = [
        "#4B0082",  # indigo
        "#800000",  # maroon
        "#0B3D91",  # deep blue
        "#145A32",  # deep green
        "#5B2C6F",  # deep purple
        "#1B2631",  # almost black
        "#7D6608",  # deep mustard
        "#512E5F",  # dark violet
        "#154360",  # dark navy
        "#641E16",  # dark red-brown
        "#0E6251",  # dark teal
    ]

    for j, c in enumerate(curves):
        s = c["s"]
        loss = c["loss"].copy()
        if use_relative:
            loss = loss - 0.5 * (loss[0] + loss[-1])

        k = int(np.argmax(loss))
        color = deep_colors[j % len(deep_colors)]

        ax.scatter(
            j, float(s[k]), float(loss[k]),
            marker="x",
            s=90,              # bigger so it stands out
            c=color,
            linewidths=3.0,    # thicker X
            depthshade=False
        )

    plt.tight_layout()
    outdir = "barrier_figs"
    os.makedirs(outdir, exist_ok=True)

    png_path = os.path.join(outdir, "barrier_surface.png")
    pdf_path = os.path.join(outdir, "barrier_surface.pdf")

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print("Saved:", png_path)
    print("Saved:", pdf_path)

if __name__ == "__main__":
    curves = load_curves("barrier_curves")
    plot_surface(
        curves,
        title="FreeMOCA: chain of connected valleys (final→final)",
        use_relative=True,
        surface_cmap="Blues_r",   # light-ish blues
        surface_alpha=0.55        # transparent/light
    )
