"""
keypoint_proposal.py
====================
Faithful re-implementation of ReKep's keypoint proposal (paper appendix A.5),
restored to replace the VLM-grounding approach in keypoint_proposal_qwen.py.

Pipeline (matches the paper):
  RGB
   -> DINOv2-with-registers (ViT-S/14)  : dense patch features
   -> bilinear upsample to image size
   -> SAM                               : object masks {m1..mn}
   -> per mask: PCA(features, 3)        : strip texture artifacts
   -> per mask: k-means (k=5)           : candidate points (median per cluster)
   -> project candidates to 3D          : via RGB-D points array (world frame)
   -> MeanShift (bandwidth 8cm)         : merge nearby duplicates
   -> filter to workspace bounds
   -> overlay numbered marks on RGB     : for the constraint VLM

DEPTH SOURCE:
  get_keypoints(rgb, points) expects `points` = [H, W, 3] world-frame XYZ,
  one 3D point per pixel. When the RGB-D camera arrives you build this from:
      depth(mm)/1000 -> backproject with intrinsics -> apply T_base_camera
  Until then, pass points=None to run the 2D half only (DINOv2 + SAM +
  clustering + numbered overlay) so you can verify candidates land on objects
  before the camera is wired in.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MeanShift
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

from transformers import AutoModel, pipeline


# ── ImageNet normalisation for DINOv2 ──────────────────────────────────────────
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
_PATCH = 16


class KeypointProposer:
    def __init__(
        self,
        device=None,
        dino_model="facebook/dinov3-vits16-pretrain-lvd1689m",
        sam_model="facebook/sam-vit-base",
        k_per_mask=5,                 # k-means clusters per mask (paper: 5)
        meanshift_bandwidth_m=0.08,   # 8 cm (paper)
        target_long_side=518,         # DINOv2 input long side (multiple of 14)
        min_mask_pixels=400,          # ignore tiny masks
        workspace_bounds=None,        # (lo[3], hi[3]) in metres, world frame
        load_sam=True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps")
        self.k = k_per_mask
        self.bandwidth_m = meanshift_bandwidth_m
        self.target_long = target_long_side
        self.min_mask_pixels = min_mask_pixels
        self.bounds = workspace_bounds

        print(f"[KP] loading DINOv2 ({dino_model}) on {self.device} ...")
        self.dino = AutoModel.from_pretrained(dino_model).to(self.device).eval()
        self.n_register = getattr(self.dino.config, "num_register_tokens", 0)

        self.sam = None
        self.sam_generator = None
        if load_sam:
            print(f"[KP] loading MobileSAM ...")
            from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
            sam = sam_model_registry["vit_t"](checkpoint="mobile_sam.pt")
            sam.to("cpu")
            sam.eval()
            self.sam_generator = SamAutomaticMaskGenerator(sam)

        self._tf = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    # ── 1. DINOv2 dense features ────────────────────────────────────────────────
    def _get_features(self, rgb):
        """rgb [H,W,3] uint8 -> features [H,W,D] float32 (upsampled)."""
        H, W = rgb.shape[:2]
        scale = self.target_long / max(H, W)
        h = max(_PATCH, int(round(H * scale)) // _PATCH * _PATCH)
        w = max(_PATCH, int(round(W * scale)) // _PATCH * _PATCH)

        img = Image.fromarray(rgb).resize((w, h), Image.BILINEAR)
        x = self._tf(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.dino(x).last_hidden_state[0]   # [1 + R + N, D]

        n_skip = 1 + self.n_register                  # CLS + register tokens
        patch_tokens = out[n_skip:]                   # [N, D]
        gh, gw = h // _PATCH, w // _PATCH
        feat = patch_tokens.reshape(gh, gw, -1).permute(2, 0, 1).unsqueeze(0)  # [1,D,gh,gw]
        feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)
        return feat[0].permute(1, 2, 0).float().cpu().numpy()   # [H,W,D]

    # ── 2. SAM masks ─────────────────────────────────────────────────────────────
    def _get_masks(self, rgb):
        if self.sam_generator is None:
            raise RuntimeError("SAM not loaded (load_sam=False)")
        results = self.sam_generator.generate(rgb)
        masks = []
        for r in results:
            m = np.asarray(r["segmentation"]).astype(bool)
            if m.sum() >= self.min_mask_pixels:
                masks.append(m)
        return masks

    # ── 3. per-mask PCA + k-means -> candidate pixels ────────────────────────────
    def _cluster(self, features, masks):
        """-> list of (py, px, mask_id)."""
        candidates = []
        for mid, m in enumerate(masks):
            ys, xs = np.where(m)
            if len(ys) < self.k:
                continue
            feats = features[ys, xs]                  # [P, D]

            # PCA to 3 dims removes texture/artifact detail (paper)
            if feats.shape[0] > 3 and feats.shape[1] > 3:
                feats = PCA(n_components=3).fit_transform(feats)

            k = min(self.k, feats.shape[0])
            km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(feats)

            for c in range(k):
                idx = np.where(km.labels_ == c)[0]
                if len(idx) == 0:
                    continue
                # median pixel position of the cluster = candidate
                cy = int(np.median(ys[idx]))
                cx = int(np.median(xs[idx]))
                candidates.append((cy, cx, mid))
        return candidates

    # ── 4. project to 3D via RGB-D points array ──────────────────────────────────
    def _project(self, candidates, points):
        """candidates [(py,px,mid)] + points [H,W,3] -> (pts3d [M,3], px [M,2], mids [M])."""
        pts3d, px, mids = [], [], []
        for (py, px_, mid) in candidates:
            p = points[py, px_]
            if p is None or np.any(~np.isfinite(p)):
                continue
            pts3d.append(p)
            px.append((px_, py))
            mids.append(mid)
        return np.array(pts3d), np.array(px), np.array(mids)

    # ── 5. workspace filter + MeanShift merge in 3D ──────────────────────────────
    def _merge(self, pts3d, px, mids):
        if len(pts3d) == 0:
            return pts3d, px, mids

        if self.bounds is not None:
            lo, hi = np.asarray(self.bounds[0]), np.asarray(self.bounds[1])
            keep = np.all((pts3d >= lo) & (pts3d <= hi), axis=1)
            pts3d, px, mids = pts3d[keep], px[keep], mids[keep]
            if len(pts3d) == 0:
                return pts3d, px, mids

        ms = MeanShift(bandwidth=self.bandwidth_m, bin_seeding=True).fit(pts3d)
        labels = ms.labels_
        merged_3d, merged_px, merged_mid = [], [], []
        for lab in np.unique(labels):
            idx = np.where(labels == lab)[0]
            merged_3d.append(pts3d[idx].mean(axis=0))
            # representative pixel + dominant mask id for this merged keypoint
            merged_px.append(px[idx][len(idx) // 2])
            merged_mid.append(np.bincount(mids[idx]).argmax())
        return np.array(merged_3d), np.array(merged_px), np.array(merged_mid)

    # ── 6. numbered overlay for the VLM ──────────────────────────────────────────
    @staticmethod
    def _overlay(rgb, pixels):
        img = Image.fromarray(rgb.copy())
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
        except Exception:
            font = ImageFont.load_default()
        for i, (x, y) in enumerate(pixels):
            r = 10
            d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 0, 0), outline=(255, 255, 255), width=2)
            d.text((x + r + 2, y - r), str(i), fill=(255, 255, 0), font=font)
        return np.array(img)

    # ── orchestration ────────────────────────────────────────────────────────────
    def get_keypoints(self, rgb, points=None, visualize=True):
        """
        rgb    : [H,W,3] uint8
        points : [H,W,3] world-frame XYZ per pixel (RGB-D). None -> 2D-only test.

        Returns dict:
          keypoints_3d : [K,3] world coords  (None if points is None)
          pixels       : [K,2] (x,y) pixel positions
          mask_ids     : [K]   which SAM mask each keypoint came from (rigidity)
          overlay      : [H,W,3] image with numbered marks
        """
        rgb = np.asarray(rgb)
        # resize to manageable size before processing
        MAX_SIDE = 800
        rgb_orig = rgb.copy()
        H0, W0 = rgb.shape[:2]
        if max(H0, W0) > MAX_SIDE:
            scale = MAX_SIDE / max(H0, W0)
            new_h = int(H0 * scale)
            new_w = int(W0 * scale)
            rgb = np.array(Image.fromarray(rgb).resize((new_w, new_h), Image.BILINEAR))
            print(f"[KP] resized image {W0}x{H0} → {new_w}x{new_h}")
        feats = self._get_features(rgb)
        masks = self._get_masks(rgb)
        print(f"[KP] {len(masks)} masks above size threshold")

        candidates = self._cluster(feats, masks)
        print(f"[KP] {len(candidates)} raw candidates from clustering")

        if points is None:
            # 2D-only mode: dedup in pixel space so the overlay isn't cluttered
            px = np.array([(cx, cy) for (cy, cx, _) in candidates])
            mids = np.array([mid for (_, _, mid) in candidates])
            if len(px) > 1:
                ms = MeanShift(bandwidth=25, bin_seeding=True).fit(px.astype(float))
                keep_px, keep_mid = [], []
                for lab in np.unique(ms.labels_):
                    idx = np.where(ms.labels_ == lab)[0]
                    keep_px.append(px[idx].mean(axis=0).astype(int))
                    keep_mid.append(np.bincount(mids[idx]).argmax())
                px, mids = np.array(keep_px), np.array(keep_mid)
            overlay = self._overlay(rgb, px) if visualize else None
            print(f"[KP] {len(px)} candidates (2D-only; no 3D until RGB-D)")
            return {"keypoints_3d": None, "pixels": px, "mask_ids": mids, "overlay": overlay}

        pts3d, px, mids = self._project(candidates, points)
        pts3d, px, mids = self._merge(pts3d, px, mids)
        overlay = self._overlay(rgb, px) if visualize else None
        print(f"[KP] {len(pts3d)} final keypoints after merge + filter")
        return {"keypoints_3d": pts3d, "pixels": px, "mask_ids": mids, "overlay": overlay}


# ── standalone 2D test (no RGB-D needed) ───────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="keypoint_candidates.jpg")
    args = ap.parse_args()

    rgb = np.array(Image.open(args.image).convert("RGB"))
    kp = KeypointProposer()
    res = kp.get_keypoints(rgb, points=None, visualize=True)
    Image.fromarray(res["overlay"]).save(args.out)
    print(f"[KP] wrote {args.out} with {len(res['pixels'])} numbered candidates")