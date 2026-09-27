"""Shared point renderer for matched PhysGaussian/Representation ficus videos."""

from pathlib import Path

import cv2
import numpy as np


SH_C0 = 0.28209479177387814


def load_ply_colors(path, opacity_threshold=0.02):
    with open(path, "rb") as stream:
        properties = []
        vertex_count = None
        while True:
            line = stream.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("property"):
                properties.append(line.split()[-1])
            elif line == "end_header":
                break
        values = np.frombuffer(
            stream.read(vertex_count * len(properties) * 4), dtype="<f4"
        ).reshape(vertex_count, len(properties))

    index = {name: i for i, name in enumerate(properties)}
    opacity = 1.0 / (1.0 + np.exp(-values[:, index["opacity"]]))
    mask = opacity > opacity_threshold
    dc = values[mask][:, [index["f_dc_0"], index["f_dc_1"], index["f_dc_2"]]]
    rgb = np.clip(0.5 + SH_C0 * dc, 0.0, 1.0)
    return (rgb[:, ::-1] * 255.0).astype(np.uint8)


class MatchedFicusRenderer:
    def __init__(self, ply_path, output, fps=30.0, size=800):
        self.colors = load_ply_colors(ply_path)
        self.size = int(size)
        self.eye = np.array([3.35, -3.35, 2.25], np.float32)
        self.target = np.array([1.0, 1.0, 0.95], np.float32)
        self.up = np.array([0.0, 0.0, 1.0], np.float32)
        forward = self.target - self.eye
        self.forward = forward / np.linalg.norm(forward)
        self.right = np.cross(self.forward, self.up)
        self.right /= np.linalg.norm(self.right)
        self.camera_up = np.cross(self.right, self.forward)
        self.focal = 0.5 * self.size / np.tan(np.deg2rad(42.0) * 0.5)

        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(output), fourcc, float(fps), (self.size, self.size)
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {output}")

    def project(self, points):
        rel = points - self.eye
        depth = rel @ self.forward
        px = self.focal * (rel @ self.right) / depth + 0.5 * self.size
        py = 0.5 * self.size - self.focal * (rel @ self.camera_up) / depth
        return px, py, depth

    def _draw_floor(self, image):
        values = np.linspace(0.0, 2.0, 9)
        for value in values:
            for start, end in (
                ([value, 0.0, 0.12], [value, 2.0, 0.12]),
                ([0.0, value, 0.12], [2.0, value, 0.12]),
            ):
                points = np.asarray([start, end], np.float32)
                px, py, depth = self.project(points)
                if np.all(depth > 0):
                    cv2.line(
                        image,
                        (int(px[0]), int(py[0])),
                        (int(px[1]), int(py[1])),
                        (205, 205, 205),
                        1,
                        cv2.LINE_AA,
                    )

    def write(self, positions, label, frame):
        image = np.full((self.size, self.size, 3), 248, np.uint8)
        self._draw_floor(image)
        px, py, depth = self.project(positions)
        valid = (
            (depth > 0.0)
            & (px >= 1)
            & (px < self.size - 1)
            & (py >= 1)
            & (py < self.size - 1)
        )
        ids = np.flatnonzero(valid)
        order = ids[np.argsort(depth[ids])[::-1]]
        x = px[order].astype(np.int32)
        y = py[order].astype(np.int32)
        color = self.colors[order]
        for ox, oy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            image[y + oy, x + ox] = color
        cv2.putText(
            image, label, (24, 38), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (25, 25, 25), 2, cv2.LINE_AA,
        )
        cv2.putText(
            image, f"t = {frame * 0.008:.3f} s", (24, 72),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (55, 55, 55), 1, cv2.LINE_AA,
        )
        self.writer.write(image)

    def close(self):
        self.writer.release()
