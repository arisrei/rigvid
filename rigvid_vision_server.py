"""
RigVid Vision API Server.

This FastAPI server provides endpoints for rigvid's vision models:
- RollingDepth (video depth estimation)
- FoundationPose video tracking (registration + tracking)

Run with: uv run --no-sync uvicorn rigvid_vision_server:app --host 0.0.0.0 --port 8766
"""

import os
import sys
import io
import base64
import tempfile
import shutil
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import cv2

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add rigvid to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

app = FastAPI(title="RigVid Vision API", version="1.0.0")

# Global model instances (lazy loaded)
_depth_predictor = None
_pose_predictor_cache = {}


def get_depth_predictor():
    """Lazy load RollingDepth predictor."""
    global _depth_predictor
    if _depth_predictor is None:
        logger.info("Loading RollingDepth model...")
        from depth_predictor import DepthPredictor
        _depth_predictor = DepthPredictor(
            checkpoint="prs-eth/rollingdepth-v1-0",
            dtype="fp16",
            color_maps=["Greys_r"],
        )
        logger.info("RollingDepth loaded successfully")
    return _depth_predictor


# ============== Request/Response Models ==============

class DepthPredictionRequest(BaseModel):
    """Request for depth prediction on a video."""
    video_b64: str  # Base64 encoded video file
    gt_depth_b64: Optional[str] = None  # Optional ground truth depth for first frame (uint16 PNG)
    mask_b64: Optional[str] = None  # Optional mask for scale calibration
    width: int = 1280
    height: int = 720


class DepthPredictionResponse(BaseModel):
    """Response from depth prediction."""
    depth_frames_b64: list[str]  # List of base64 encoded depth frames (uint16 PNG)
    alpha: Optional[float] = None  # Scale factor if calibrated
    beta: Optional[float] = None  # Shift factor if calibrated


class PoseRolloutRequest(BaseModel):
    """Request for FoundationPose video tracking."""
    video_b64: str  # Base64 encoded video file (or folder of frames)
    mesh_obj_b64: str  # Base64 encoded .obj mesh file
    mask_b64: str  # Base64 encoded mask for first frame
    intrinsics: list[list[float]]  # 3x3 camera intrinsics matrix
    depth_frames_b64: list[str]  # List of base64 encoded depth frames
    est_refine_iter: int = 10
    track_refine_iter: int = 4
    debug: int = 2


class PoseRolloutResponse(BaseModel):
    """Response from pose rollout."""
    poses: list[list[list[float]]]  # List of 4x4 transformation matrices
    success: bool
    tracking_video_b64: Optional[str] = None  # Visualization video


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: dict


# ============== Utility Functions ==============

def b64_to_bytes(b64_str: str) -> bytes:
    """Decode base64 string to bytes."""
    return base64.b64decode(b64_str)


def bytes_to_b64(data: bytes) -> str:
    """Encode bytes to base64 string."""
    return base64.b64encode(data).decode("utf-8")


def b64_to_image(b64_str: str) -> np.ndarray:
    """Decode base64 string to numpy array."""
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_bytes))
    return np.array(img)


def image_to_b64(img: np.ndarray) -> str:
    """Encode numpy array to base64 PNG string."""
    pil_img = Image.fromarray(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def file_to_b64(filepath: str) -> str:
    """Read file and encode to base64."""
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def b64_to_file(b64_str: str, filepath: str):
    """Decode base64 and write to file."""
    with open(filepath, "wb") as f:
        f.write(base64.b64decode(b64_str))


# ============== API Endpoints ==============

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check server health and loaded models."""
    return HealthResponse(
        status="healthy",
        models_loaded={
            "depth_predictor": _depth_predictor is not None,
        }
    )


@app.post("/depth_prediction", response_model=DepthPredictionResponse)
async def run_depth_prediction(request: DepthPredictionRequest):
    """Run RollingDepth on a video."""
    try:
        logger.info("Depth prediction request received")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save input video
            video_path = os.path.join(tmpdir, "input.mp4")
            b64_to_file(request.video_b64, video_path)
            
            # Run depth prediction
            predictor = get_depth_predictor()
            depth_npy = predictor.predict(
                input_video_path=video_path,
                output_dir=tmpdir,
            )

            # Optionally calibrate with ground truth depth
            alpha, beta = None, None
            if request.gt_depth_b64 and request.mask_b64:
                from utils import find_center_of_mask, find_scale_and_shift

                # Save mask temporarily
                mask_path = os.path.join(tmpdir, "mask.png")
                b64_to_file(request.mask_b64, mask_path)

                # Load ground truth depth
                gt_depth = b64_to_image(request.gt_depth_b64)

                # Find scale and shift
                pixel_coords = find_center_of_mask(mask_path, window_size=20)
                alpha, beta = find_scale_and_shift(depth_npy, gt_depth, pixel_coords, mask_invalid=True)

                # Apply calibration
                calibrated_depth = []
                for frame in depth_npy:
                    d = cv2.resize(frame.astype(np.float32), (request.width, request.height), interpolation=cv2.INTER_CUBIC)
                    d = alpha * d + beta
                    d[d < 0] = 0
                    calibrated_depth.append(d.astype(np.uint16))
                depth_npy = np.array(calibrated_depth)
            else:
                # Just resize without calibration
                calibrated_depth = []
                for frame in depth_npy:
                    d = cv2.resize(frame.astype(np.float32), (request.width, request.height), interpolation=cv2.INTER_CUBIC)
                    # Scale to uint16 range (assuming depth is in some normalized range)
                    d = (d * 1000).astype(np.uint16)  # Convert to mm
                    calibrated_depth.append(d)
                depth_npy = np.array(calibrated_depth)

            # Encode depth frames
            depth_frames_b64 = []
            for frame in depth_npy:
                depth_frames_b64.append(image_to_b64(frame))

            return DepthPredictionResponse(
                depth_frames_b64=depth_frames_b64,
                alpha=alpha,
                beta=beta,
            )
    except Exception as e:
        logger.error(f"Depth prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pose_rollout", response_model=PoseRolloutResponse)
async def run_pose_rollout(request: PoseRolloutRequest):
    """Run FoundationPose registration + tracking on video frames."""
    try:
        logger.info("Pose rollout request received")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create folder structure expected by YcbineoatReader
            # Structure: tmpdir/rgb/*.png, tmpdir/depth/*.png, tmpdir/masks/*.png, tmpdir/cam_K.txt
            rgb_dir = os.path.join(tmpdir, "rgb")
            depth_dir = os.path.join(tmpdir, "depth")
            masks_dir = os.path.join(tmpdir, "masks")
            os.makedirs(rgb_dir, exist_ok=True)
            os.makedirs(depth_dir, exist_ok=True)
            os.makedirs(masks_dir, exist_ok=True)

            # Save video and extract frames
            video_path = os.path.join(tmpdir, "input.mp4")
            b64_to_file(request.video_b64, video_path)

            # Extract frames from video
            cap = cv2.VideoCapture(video_path)
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join(rgb_dir, f"{frame_idx:06d}.png"), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                frame_idx += 1
            cap.release()

            # Save depth frames
            for i, depth_b64 in enumerate(request.depth_frames_b64):
                depth_img = b64_to_image(depth_b64)
                cv2.imwrite(os.path.join(depth_dir, f"{i:06d}.png"), depth_img)

            # Save mask (only first frame needed for registration)
            mask_img = b64_to_image(request.mask_b64)
            cv2.imwrite(os.path.join(masks_dir, "000000.png"), mask_img)

            # Save intrinsics
            intrinsics = np.array(request.intrinsics)
            np.savetxt(os.path.join(tmpdir, "cam_K.txt"), intrinsics.flatten())

            # Save mesh
            mesh_path = os.path.join(tmpdir, "mesh.obj")
            b64_to_file(request.mesh_obj_b64, mesh_path)

            # Run pose rollout
            debug_dir = os.path.join(tmpdir, "fp_outputs")
            from pose_rollout_predictor import PoseRolloutPredictor

            pose_predictor = PoseRolloutPredictor(
                data_path=tmpdir,
                mesh_file=mesh_path,
                est_refine_iter=request.est_refine_iter,
                track_refine_iter=request.track_refine_iter,
                debug=request.debug,
                debug_dir=debug_dir,
            )
            pose_predictor.run()

            # Load poses
            poses = []
            pose_dir = os.path.join(debug_dir, "ob_in_cam")
            for pose_file in sorted(Path(pose_dir).glob("*.txt")):
                pose = np.loadtxt(pose_file).reshape(4, 4)
                poses.append(pose.tolist())

            # Load tracking video if available
            tracking_video_b64 = None
            tracking_video_path = os.path.join(debug_dir, "tracking_video.mp4")
            if os.path.exists(tracking_video_path):
                tracking_video_b64 = file_to_b64(tracking_video_path)

            return PoseRolloutResponse(
                poses=poses,
                success=len(poses) > 0,
                tracking_video_b64=tracking_video_b64,
            )
    except Exception as e:
        logger.error(f"Pose rollout error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9766)

