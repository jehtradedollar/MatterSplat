"""Read point clouds and images from E57 files (tested on Matterport Pro3 exports).

Pose conventions (kept exactly as stored in the E57, nothing converted yet):

* Every pose is ``rotation`` (unit quaternion, stored and returned as w, x, y, z)
  plus ``translation`` (x, y, z, metres). It maps the object's local frame into
  the file's world frame::

      p_world = R(q) @ p_local + t

* data3D points are stored in the scanner's local frame; :meth:`E57File.read_scan`
  applies the scan pose when ``world=True`` (the default).

* images2D poses map the *camera* frame into world. For pinholeRepresentation the
  camera frame follows the OpenGL convention: the camera looks down **-Z**,
  +X points right in the image and +Y points up. Pixel (0, 0) is the top-left
  corner, u grows to the right and v grows downwards, so::

      p_cam = R(q).T @ (p_world - t)
      f_px  = focalLength / pixelWidth
      u = principalPointX + f_px * p_cam.x / -p_cam.z
      v = principalPointY - f_px * p_cam.y / -p_cam.z

  This was confirmed by reprojecting the coloured Matterport points into their
  skybox faces (mean colour error ~1-2 DN with this convention, 55-90 DN for
  every other sign combination). To get an OpenCV/COLMAP camera (+Z forward,
  +Y down) flip the camera's Y and Z axes: R_cv = R(q) @ diag(1, -1, -1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pye57
from pye57 import libe57

REPRESENTATIONS = (
    "pinholeRepresentation",
    "sphericalRepresentation",
    "cylindricalRepresentation",
    "visualReferenceRepresentation",
)
BLOB_KEYS = ("jpegImage", "pngImage", "imageMask")

POSE_CONVENTION = (
    "p_world = R(q) @ p_local + t, q stored as (w, x, y, z). Scan points are local to the scan pose. "
    "Pinhole cameras: OpenGL axes (look down -Z, +X right, +Y up); "
    "u = cx + f*x/-z, v = cy - f*y/-z, f = focalLength/pixelWidth, pixel (0,0) top-left."
)

_NODE_CLASSES = {
    libe57.E57_STRUCTURE: libe57.StructureNode,
    libe57.E57_VECTOR: libe57.VectorNode,
    libe57.E57_COMPRESSED_VECTOR: libe57.CompressedVectorNode,
    libe57.E57_INTEGER: libe57.IntegerNode,
    libe57.E57_SCALED_INTEGER: libe57.ScaledIntegerNode,
    libe57.E57_FLOAT: libe57.FloatNode,
    libe57.E57_STRING: libe57.StringNode,
    libe57.E57_BLOB: libe57.BlobNode,
}


def _cast(node):
    """libe57 hands back generic Nodes from get(i); downcast to the concrete type."""
    if type(node) is libe57.Node:
        return _NODE_CLASSES[node.type()](node)
    return node


def _children(struct) -> dict:
    struct = _cast(struct)
    return {c.elementName(): c for c in (_cast(struct.get(i)) for i in range(struct.childCount()))}


def _scalar(node):
    if isinstance(node, libe57.ScaledIntegerNode):
        return node.scaledValue()
    if isinstance(node, (libe57.IntegerNode, libe57.FloatNode, libe57.StringNode)):
        return node.value()
    return None


def _read_pose(node) -> tuple[np.ndarray, np.ndarray]:
    """Return (quaternion wxyz, translation xyz); identity if absent."""
    q = np.array([1.0, 0.0, 0.0, 0.0])
    t = np.zeros(3)
    if node is None:
        return q, t
    pose = _children(node)
    if "rotation" in pose:
        r = _children(pose["rotation"])
        q = np.array([_scalar(r[k]) for k in "wxyz"], dtype=float)
    if "translation" in pose:
        tr = _children(pose["translation"])
        t = np.array([_scalar(tr[k]) for k in "xyz"], dtype=float)
    return q, t


def quat_to_matrix(q) -> np.ndarray:
    """3x3 rotation matrix from a (w, x, y, z) quaternion (normalised first)."""
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


@dataclass
class ScanInfo:
    index: int
    guid: str
    name: str
    point_count: int
    fields: list[str]
    rotation: np.ndarray  # w, x, y, z
    translation: np.ndarray
    metadata: dict = field(default_factory=dict)  # sensor*, description, bounds, ...


@dataclass
class ImageInfo:
    index: int
    guid: str
    name: str
    scan_guid: str | None  # associatedData3DGuid
    representation: str  # e.g. "pinholeRepresentation"
    width: int | None
    height: int | None
    intrinsics: dict  # every scalar child of the representation node, as stored
    blob_format: str | None  # "jpegImage" or "pngImage"
    blob_size: int
    rotation: np.ndarray  # w, x, y, z (camera -> world, see module docstring)
    translation: np.ndarray

    @property
    def focal_px(self) -> float | None:
        """Pinhole focal length in pixels (focalLength / pixelWidth)."""
        f, pw = self.intrinsics.get("focalLength"), self.intrinsics.get("pixelWidth")
        return f / pw if f and pw else None


class E57File:
    """Lazy reader: metadata is parsed up front, points and image blobs on demand."""

    def __init__(self, path: str):
        self.path = path
        self._e57 = pye57.E57(path)
        root = _children(self._e57.root)
        self.guid = _scalar(root.get("guid")) if "guid" in root else None
        self.scans = self._parse_scans(root.get("data3D"))
        self._image_nodes: list = []
        self.images = self._parse_images(root.get("images2D"))

    def close(self):
        self._e57.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _parse_scans(self, data3d) -> list[ScanInfo]:
        scans = []
        for i in range(data3d.childCount() if data3d is not None else 0):
            c = _children(_cast(data3d.get(i)))
            points = c["points"]
            proto = _children(points.prototype())
            q, t = _read_pose(c.get("pose"))
            meta = {k: _scalar(v) for k, v in c.items() if _scalar(v) is not None}
            for k in ("cartesianBounds", "sphericalBounds", "indexBounds", "intensityLimits", "colorLimits"):
                if k in c:
                    meta[k] = {kk: _scalar(vv) for kk, vv in _children(c[k]).items()}
            scans.append(ScanInfo(
                index=i,
                guid=meta.pop("guid", ""),
                name=meta.pop("name", f"Scan {i}"),
                point_count=points.childCount(),
                fields=list(proto),
                rotation=q,
                translation=t,
                metadata=meta,
            ))
        return scans

    def _parse_images(self, images2d) -> list[ImageInfo]:
        images = []
        for i in range(images2d.childCount() if images2d is not None else 0):
            c = _children(_cast(images2d.get(i)))
            rep_name = next((r for r in REPRESENTATIONS if r in c), None)
            intrinsics, blob_key, blob_size = {}, None, 0
            if rep_name:
                rep = _children(c[rep_name])
                intrinsics = {k: _scalar(v) for k, v in rep.items() if _scalar(v) is not None}
                blob_key = next((k for k in ("jpegImage", "pngImage") if k in rep), None)
                blob_size = rep[blob_key].byteCount() if blob_key else 0
                self._image_nodes.append(rep)
            else:
                self._image_nodes.append(None)
            q, t = _read_pose(c.get("pose"))
            images.append(ImageInfo(
                index=i,
                guid=_scalar(c["guid"]) if "guid" in c else "",
                name=_scalar(c["name"]) if "name" in c else f"Image {i}",
                scan_guid=_scalar(c["associatedData3DGuid"]) if "associatedData3DGuid" in c else None,
                representation=rep_name or "none",
                width=intrinsics.get("imageWidth"),
                height=intrinsics.get("imageHeight"),
                intrinsics=intrinsics,
                blob_format=blob_key,
                blob_size=blob_size,
                rotation=q,
                translation=t,
            ))
        return images

    def read_image_bytes(self, index: int, key: str | None = None) -> bytes:
        """Raw JPEG/PNG bytes of an image (or its imageMask with key='imageMask')."""
        rep = self._image_nodes[index]
        key = key or self.images[index].blob_format
        if rep is None or key not in rep:
            raise KeyError(f"image {index} has no {key}")
        blob = rep[key]
        buf = np.empty(blob.byteCount(), dtype=np.uint8)
        blob.read(buf, 0, blob.byteCount())
        return buf.tobytes()

    def images_for_scan(self, scan: ScanInfo) -> list[ImageInfo]:
        return [im for im in self.images if im.scan_guid == scan.guid]

    def read_scan(self, index: int, world: bool = True, fraction: float = 1.0,
                  rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        """Valid points of one scan as (Nx3 float64, Nx3 uint8 colours or None).

        Points flagged by cartesianInvalidState != 0 are dropped. ``fraction`` < 1
        keeps a random subset (for previews).
        """
        raw = self._e57.read_scan_raw(index)
        keep = np.ones(len(raw["cartesianX"]), dtype=bool)
        if "cartesianInvalidState" in raw:
            keep &= raw["cartesianInvalidState"] == 0
        if fraction < 1.0:
            rng = rng or np.random.default_rng(index)
            keep &= rng.random(len(keep)) < fraction
        pts = np.stack([raw["cartesianX"][keep], raw["cartesianY"][keep], raw["cartesianZ"][keep]], axis=1)
        if world:
            scan = self.scans[index]
            pts = pts @ quat_to_matrix(scan.rotation).T + scan.translation
        cols = None
        if "colorRed" in raw:
            cols = np.stack([raw["colorRed"][keep], raw["colorGreen"][keep], raw["colorBlue"][keep]], axis=1)
            cols = cols.astype(np.uint8)
        return pts, cols

    def iter_points(self, fraction: float = 1.0, scans=None) -> Iterator[tuple[int, np.ndarray, np.ndarray | None]]:
        for i in scans if scans is not None else range(len(self.scans)):
            pts, cols = self.read_scan(i, fraction=fraction)
            yield i, pts, cols

    def read_points(self, fraction: float = 1.0, scans=None) -> tuple[np.ndarray, np.ndarray | None]:
        """All scans merged in the world frame. Full Matterport files hold ~10^8-10^9
        points, so pass a fraction unless you really have the RAM."""
        parts = list(self.iter_points(fraction, scans))
        pts = np.concatenate([p for _, p, _ in parts]).astype(np.float32)
        cols = None if parts[0][2] is None else np.concatenate([c for _, _, c in parts])
        return pts, cols


def project_pinhole(image: ImageInfo, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project world points into a pinhole image. Returns (uv Nx2, in_front mask)."""
    pc = (points_world - image.translation) @ quat_to_matrix(image.rotation)
    depth = -pc[:, 2]
    f = image.focal_px
    with np.errstate(divide="ignore", invalid="ignore"):
        u = image.intrinsics["principalPointX"] + f * pc[:, 0] / depth
        v = image.intrinsics["principalPointY"] - f * pc[:, 1] / depth
    return np.stack([u, v], axis=1), depth > 0


def read_e57(path: str, fraction: float = 1.0, load_images: bool = True) -> dict:
    """Convenience one-shot read: {'points', 'colors', 'scans', 'images'}.

    Each image dict carries its metadata plus ``blob`` (bytes) when load_images.
    """
    with E57File(path) as f:
        pts, cols = f.read_points(fraction)
        images = []
        for im in f.images:
            d = image_to_dict(im)
            if load_images and im.blob_format:
                d["blob"] = f.read_image_bytes(im.index)
            images.append(d)
        return {"points": pts, "colors": cols, "scans": [scan_to_dict(s) for s in f.scans], "images": images}


def scan_to_dict(s: ScanInfo) -> dict:
    return {
        "index": s.index, "guid": s.guid, "name": s.name, "point_count": s.point_count,
        "fields": s.fields, "rotation_wxyz": s.rotation.tolist(), "translation": s.translation.tolist(),
        **s.metadata,
    }


def image_to_dict(im: ImageInfo) -> dict:
    return {
        "index": im.index, "guid": im.guid, "name": im.name, "scan_guid": im.scan_guid,
        "representation": im.representation, "width": im.width, "height": im.height,
        "intrinsics": im.intrinsics, "focal_px": im.focal_px, "blob_format": im.blob_format,
        "blob_size": im.blob_size, "rotation_wxyz": im.rotation.tolist(), "translation": im.translation.tolist(),
    }
