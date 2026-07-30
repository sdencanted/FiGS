# Developed from: https://github.com/madang6/flightroom_ns_process/tree/feature/video_process

import multiprocessing
from pathlib import Path
import json
import shutil
from typing import List, Tuple, Dict, Union, Optional

from nerfstudio.process_data.images_to_nerfstudio_dataset import (
    ImagesToNerfstudioDataset,
)

import figs.utilities.capture_helper as ch
import cv2
import numpy as np
import open3d as o3d
import subprocess
import os

import pycolmap
from tqdm.auto import tqdm

import logging
import re


logger = logging.getLogger("figs")


def get_camera_images(image_dir: Path, camera_pattern: str = r"_Cam(\d+)\.") -> Dict[str, List[Path]]:
    """
    Group images by camera from a multi-camera folder.
    
    Args:
        image_dir: Directory containing images
        camera_pattern: Regex pattern to extract camera ID from filename.
                      Should capture the camera number in a group.
    
    Returns:
        Dictionary mapping camera_id (str like "Cam01") to list of image paths
    """
    camera_images = {}
    image_extensions = (".png", ".jpg", ".jpeg")
    
    for img_path in image_dir.iterdir():
        if img_path.suffix.lower() not in image_extensions:
            continue
        
        # Try to match camera pattern in filename
        match = re.search(camera_pattern, img_path.name)
        if match:
            camera_id = f"Cam{match.group(1).zfill(2)}"
            if camera_id not in camera_images:
                camera_images[camera_id] = []
            camera_images[camera_id].append(img_path)
        else:
            # If no camera pattern found, treat as single camera
            if "Cam01" not in camera_images:
                camera_images["Cam01"] = []
            camera_images["Cam01"].append(img_path)
    
    # Sort images by filename for consistent ordering
    for camera_id in camera_images:
        camera_images[camera_id].sort(key=lambda x: x.name)
    
    return camera_images


def is_multi_camera_folder(images_path: Path, camera_pattern: str = r"_Cam(\d+)\.") -> bool:
    """
    Check if the folder contains images from multiple cameras.
    
    Args:
        images_path: Path to the images directory
        camera_pattern: Regex pattern to detect camera IDs
    
    Returns:
        True if multiple cameras detected, False otherwise
    """
    camera_images = get_camera_images(images_path, camera_pattern)
    return len(camera_images) > 1


def process_single_camera(
    camera_id: str,
    images: List[Path],
    process_path: Path,
    sfm_path: Path,
    camera_config: Dict,
    extractor_config: Dict,
    capture_cfg_name: str,
    gsplats_path: Path,
    config_path: Path,
    workspace_path: Path,
    outputs_path: Path
) -> Tuple[Path, Path]:
    """
    Process a single camera's images through the full pipeline.
    
    Args:
        camera_id: Camera identifier (e.g., "Cam01")
        images: List of image paths for this camera
        process_path: Base process path for the scene
        sfm_path: SfM output path
        camera_config: Camera configuration
        extractor_config: Extractor configuration
        capture_cfg_name: Name of the capture config
        gsplats_path: Path to gsplats directory
        config_path: Path to configs directory
        workspace_path: Path to workspace directory
        outputs_path: Path to outputs directory
    
    Returns:
        Tuple of (sparse_pc_path, transforms_path) for this camera
    """
    logger.info(f"Processing camera: {camera_id}")
    
    # Create camera-specific subdirectory
    camera_sfm_path = sfm_path / camera_id
    camera_sfm_path.mkdir(parents=True, exist_ok=True)
    
    # Create temporary directory with camera images
    camera_images_dir = process_path / camera_id
    camera_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy images to camera directory
    for img_path in images:
        dest_path = camera_images_dir / img_path.name
        shutil.copy2(img_path, dest_path)
    
    # Run ns_process for this camera
    ns_obj = ImagesToNerfstudioDataset(
        data=camera_images_dir,
        output_dir=camera_sfm_path,
        camera_type="perspective",
        matching_method="sequential",
        sfm_tool="hloc",
        gpu=True,
        matcher_type="superpoint+lightglue",
        use_single_camera_mode=False
    )
    ns_obj.main()
    
    # Load transforms and sparse point cloud
    camera_tfm_path = camera_sfm_path / "transforms.json"
    camera_spc_path = camera_sfm_path / "sparse_pc.ply"
    
    with open(camera_tfm_path, "r") as f:
        camera_tfm_data = json.load(f)
    
    sparse_pcloud = o3d.io.read_point_cloud(camera_spc_path.as_posix())
    
    # Use sfm config if camera config is not provided
    if camera_config is None:
        fx, fy = camera_tfm_data["fl_x"], camera_tfm_data["fl_y"]
        cx, cy = camera_tfm_data["cx"], camera_tfm_data["cy"]
        k1, k2 = camera_tfm_data["k1"], camera_tfm_data["k2"]
        p1, p2 = camera_tfm_data["p1"], camera_tfm_data["p2"]
        
        camera_config = {
            "model": camera_tfm_data["camera_model"],
            "height": camera_tfm_data["h"],
            "width": camera_tfm_data["w"],
            "intrinsics_matrix": [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0]
            ],
            "distortion_coefficients": [k1, k2, p1, p2]
        }
    
    # Compute the transform using aruco markers
    Psfm, Parc = extract_positions(camera_sfm_path, extractor_config, camera_config)
    cs, Rs, ts = ch.compute_ransac_transform(Psfm, Parc)
    
    # Transform camera poses
    for frame in camera_tfm_data["frames"]:
        Tc2s = np.array(frame["transform_matrix"])
        Tc2w = np.eye(4)
        Tc2w[:3, :3], Tc2w[:3, 3] = Rs @ Tc2s[:3, :3], cs * Rs @ Tc2s[:3, 3] + ts
        frame["transform_matrix"] = Tc2w.tolist()
    
    # Transform sparse points
    sparse_points = np.asarray(sparse_pcloud.points)
    for idx, point in enumerate(sparse_points):
        sparse_points[idx, :] = cs * Rs @ point + ts
    
    sparse_pcloud.points = o3d.utility.Vector3dVector(sparse_points)
    
    # Save transformed files
    camera_output_tfm_path = camera_sfm_path / "transforms.json"
    camera_output_spc_path = camera_sfm_path / "sparse_pc.ply"
    
    with open(camera_output_tfm_path, "w", encoding="utf8") as f:
        json.dump(camera_tfm_data, f, indent=4)
    
    o3d.io.write_point_cloud(camera_output_spc_path.as_posix(), sparse_pcloud)
    
    return camera_output_spc_path, camera_output_tfm_path


def merge_multi_camera_results(
    camera_results: Dict[str, Tuple[Path, Path]],
    output_sfm_path: Path,
    camera_config: Dict
) -> Tuple[Path, Path]:
    """
    Merge results from multiple cameras into a single reconstruction.
    
    Args:
        camera_results: Dictionary mapping camera_id to (sparse_pc_path, transforms_path)
        output_sfm_path: Path to save merged results
        camera_config: Camera configuration (shared intrinsics)
    
    Returns:
        Tuple of (merged_sparse_pc_path, merged_transforms_path)
    """
    logger.info("Merging multi-camera results...")
    
    # Collect all transforms and points
    all_frames = []
    all_points = []
    
    for camera_id, (spc_path, tfm_path) in camera_results.items():
        # Load transforms
        with open(tfm_path, "r") as f:
            tfm_data = json.load(f)
        
        # Update file paths to be relative to merged output
        for frame in tfm_data["frames"]:
            # Extract original filename and make path relative
            original_name = Path(frame["file_path"]).name
            # The images are stored in sfm/{camera_id}/images/{original_name}
            # but the transforms.json file_path should be relative to the workspace
            # So we use {camera_id}/{original_name} which will be relative to sfm/
            frame["file_path"] = f"{camera_id}/images/{original_name}"
            all_frames.append(frame)
        
        # Load and collect points
        pcloud = o3d.io.read_point_cloud(spc_path.as_posix())
        points = np.asarray(pcloud.points)
        if len(points) > 0:
            all_points.append(points)
    
    # Merge all points
    if all_points:
        merged_points = np.vstack(all_points)
    else:
        merged_points = np.array([]).reshape(0, 3)
    
    # Create merged transforms.json
    merged_tfm_data = {
        "camera_model": camera_config["model"],
        "h": camera_config["height"],
        "w": camera_config["width"],
        "fl_x": camera_config["intrinsics_matrix"][0][0],
        "fl_y": camera_config["intrinsics_matrix"][1][1],
        "cx": camera_config["intrinsics_matrix"][0][2],
        "cy": camera_config["intrinsics_matrix"][1][2],
        "k1": camera_config["distortion_coefficients"][0] if len(camera_config["distortion_coefficients"]) > 0 else 0,
        "k2": camera_config["distortion_coefficients"][1] if len(camera_config["distortion_coefficients"]) > 1 else 0,
        "p1": camera_config["distortion_coefficients"][2] if len(camera_config["distortion_coefficients"]) > 2 else 0,
        "p2": camera_config["distortion_coefficients"][3] if len(camera_config["distortion_coefficients"]) > 3 else 0,
        "frames": all_frames
    }
    
    # Save merged results
    output_sfm_path.mkdir(parents=True, exist_ok=True)
    merged_tfm_path = output_sfm_path / "transforms.json"
    merged_spc_path = output_sfm_path / "sparse_pc.ply"
    
    with open(merged_tfm_path, "w", encoding="utf8") as f:
        json.dump(merged_tfm_data, f, indent=4)
    
    # Save merged point cloud
    merged_pcloud = o3d.geometry.PointCloud()
    merged_pcloud.points = o3d.utility.Vector3dVector(merged_points)
    o3d.io.write_point_cloud(merged_spc_path.as_posix(), merged_pcloud)
    
    return merged_spc_path, merged_tfm_path


def generate_gsplat(scene_file_name: str, capture_cfg_name: str = 'default',
                    gsplats_path: Optional[Path] = None, config_path: Optional[Path] = None,
                    use_images: bool = False) -> None:
    """
    Generate a Gaussian Splatting model from capture data.
    
    Args:
        scene_file_name: Name of the scene to process
        capture_cfg_name: Name of the capture configuration
        gsplats_path: Path to gsplats directory (default: auto-detected)
        config_path: Path to configs directory (default: auto-detected)
        use_images: If True, use existing images; if False, extract from video

    The capture configuration may include a ``reconstruction`` object with
    ``mode: "flat_rig"``. This uses the rig-aware Nerfstudio fork to reconstruct
    flat image names such as ``f0001-mid_Cam01.png`` as one synchronized camera
    rig. Rig processing requires ``use_images=True``. For the rig
    capture, use ``capture_cfg_name="flat_rig"``.
    """
    # Initialize base paths
    if gsplats_path is None:
        gsplats_path = Path(__file__).parent.parent.parent.parent.parent / 'gsplats'

    if config_path is None:
        config_path = Path(__file__).parent.parent.parent.parent.parent / 'configs'

    capture_cfg_path = config_path / 'captures'
    capture_path = gsplats_path / 'capture'
    workspace_path = gsplats_path / 'workspace'
    image_extensions = (".png", ".jpg")
    
    if use_images:
        # Prefer an exact capture-relative path.  Rig captures commonly have
        # variants such as ``<name>`` and ``<name>_full``; substring globbing
        # matches both and incorrectly rejects the requested rig folder.
        images_path = capture_path / scene_file_name
        if not images_path.is_dir():
            # Preserve support for callers that provide only a distinctive
            # portion of an older capture directory name.
            images = [path for path in capture_path.glob(f"*{scene_file_name}*")
                      if path.is_dir()]
            if len(images) == 0:
                raise FileNotFoundError(
                    f"No image dataset found for '{scene_file_name}' in {capture_path}"
                )
            if len(images) > 1:
                raise ValueError(
                    f"Multiple image datasets found for '{scene_file_name}' in "
                    f"{capture_path}: {images}. Pass the exact directory name."
                )
            images_path = images[0]
            logger.info(f"using {images_path}")
    else:
        # Find the correct video path
        video_files = list(capture_path.glob(f"*{scene_file_name}*"))
        if len(video_files) == 0:
            raise FileNotFoundError(f"No file found with name containing '{scene_file_name}' in {capture_path}")
        elif len(video_files) > 1:
            raise ValueError(f"Multiple files found with name containing '{scene_file_name}' in {capture_path}")
        else:
            video_path = str(video_files[0])

    # Initialize process paths
    process_path = workspace_path / scene_file_name

    spc_path = process_path / "sparse_pc.ply"
    tfm_path = process_path / "transforms.json"

    sfm_path = process_path / "sfm"
    sfm_spc_path = sfm_path / "sparse_pc.ply"
    sfm_tfm_path = sfm_path / "transforms.json"
    
    process_path.mkdir(parents=True, exist_ok=True)

    # Initialize output paths
    outputs_path = workspace_path / 'outputs'
    output_path = outputs_path / scene_file_name

    output_path.mkdir(parents=True, exist_ok=True)

    # Load the capture config
    capture_config_file = capture_cfg_path / f"{capture_cfg_name}.json"
    with open(capture_config_file, "r") as file:
        capture_configs = json.load(file)

    camera_config = capture_configs["camera"]
    extractor_config = capture_configs["extractor"]
    reconstruction_config = capture_configs.get("reconstruction", {})
    reconstruction_mode = reconstruction_config.get("mode", "single_camera")

    # Extract the frame data
    if not use_images:
        images_path = process_path / "images"
        images_path.mkdir(parents=True, exist_ok=True)
        extract_frames(video_path, images_path, extractor_config)

    if reconstruction_mode in ["flat_rig", "rig"]:
        if not use_images:
            raise ValueError("rig reconstruction requires use_images=True.")

        logger.info("Processing flat camera rig with Nerfstudio rig bundle adjustment")
        ns_obj = ImagesToNerfstudioDataset(
            data=images_path,
            output_dir=sfm_path,
            camera_type=reconstruction_config.get("camera_type", "perspective"),
            matching_method=reconstruction_config.get("matching_method", "sequential"),
            sfm_tool=reconstruction_config.get("sfm_tool", "colmap"),
            gpu=reconstruction_config.get("gpu", True),
            num_downscales=reconstruction_config.get("num_downscales", 0),
            use_single_camera_mode=False,
            use_rig=True,
            matcher_type=reconstruction_config.get("matcher_type", "superpoint+lightglue"),
            feature_type=reconstruction_config.get("feature_type", "superpoint+lightglue"),
        )
        ns_obj.main()
        shutil.copy2(sfm_spc_path, spc_path)
        shutil.copy2(sfm_tfm_path, tfm_path)

        

        # Load the resulting transforms.json and sparse_points.ply
        with open(sfm_tfm_path, "r") as f:
            tfm_data = json.load(f)
        
        sparse_pcloud = o3d.io.read_point_cloud(sfm_spc_path.as_posix())

        # Rig transforms carry calibration on each frame, since every camera has
        # its own intrinsics. Use those values when estimating marker poses.
        logger.info("Extracting camera and marker positions from rig transforms")
        Psfm, Parc = extract_positions(sfm_path, extractor_config, None)
        logger.info(f"Extracted {Psfm.shape[1]} camera positions and {Parc.shape[1]} marker positions")
        cs, Rs, ts = ch.compute_ransac_transform(Psfm, Parc)
        logger.info(f"Computed RANSAC transform: scale={cs}, rotation=\n{Rs}, translation={ts}")
        for frame in tqdm(tfm_data["frames"], desc="Updating camera transforms"):
            Tc2s = np.array(frame["transform_matrix"])

            Tc2w = np.eye(4)
            Tc2w[:3, :3] = Rs @ Tc2s[:3, :3]
            Tc2w[:3, 3] = cs * Rs @ Tc2s[:3, 3] + ts
            frame["transform_matrix"] = Tc2w.tolist()
        logger.info("Updated camera transforms with RANSAC alignment")
        sparse_points = np.asarray(sparse_pcloud.points)
        logger.info(f"Transforming {sparse_points.shape[0]} sparse points with RANSAC alignment")
        if len(sparse_points):
            logger.info("Applying RANSAC transform to sparse point cloud")
            transformed_points = (cs * (Rs @ sparse_points.T)).T + ts
            logger.info("Transformed sparse point cloud with RANSAC alignment")
            transformed_points = np.ascontiguousarray(transformed_points, dtype=np.float64)
            sparse_pcloud.points = o3d.utility.Vector3dVector(
                transformed_points
            )
        logger.info("Transformed sparse point cloud with RANSAC alignment")
        with open(tfm_path, "w", encoding="utf8") as f:
            json.dump(tfm_data, f, indent=4)
        logger.info(f"Saved updated transforms.json to {tfm_path}")
        o3d.io.write_point_cloud(spc_path.as_posix(), sparse_pcloud)
    # Check if this is a multi-camera folder
    elif is_multi_camera_folder(images_path):
        logger.info(f"Detected multi-camera folder with {len(get_camera_images(images_path))} cameras")
        
        # Process each camera separately
        camera_images = get_camera_images(images_path)
        camera_results = {}
        
        for camera_id, cam_images in camera_images.items():
            cam_spc, cam_tfm = process_single_camera(
                camera_id=camera_id,
                images=cam_images,
                process_path=process_path,
                sfm_path=sfm_path,
                camera_config=camera_config,
                extractor_config=extractor_config,
                capture_cfg_name=capture_cfg_name,
                gsplats_path=gsplats_path,
                config_path=config_path,
                workspace_path=workspace_path,
                outputs_path=outputs_path
            )
            camera_results[camera_id] = (cam_spc, cam_tfm)
        
        # Merge results
        merged_spc, merged_tfm = merge_multi_camera_results(
            camera_results, sfm_path, camera_config
        )
        
        # Copy merged results to final locations
        shutil.copy2(merged_spc, spc_path)
        shutil.copy2(merged_tfm, tfm_path)
    else:
        logger.info("Processing single camera")
        
        # Run the ns_process step
        ns_obj = ImagesToNerfstudioDataset(
            data=images_path, output_dir=sfm_path,
            camera_type="perspective", matching_method="sequential", 
            sfm_tool="hloc", gpu=True, matcher_type="superpoint+lightglue",
            use_single_camera_mode=False
        )
        ns_obj.main()

        # Load the resulting transforms.json and sparse_points.ply
        with open(sfm_tfm_path, "r") as f:
            tfm_data = json.load(f)
        
        sparse_pcloud = o3d.io.read_point_cloud(sfm_spc_path.as_posix())
        
        # Use sfm config if camera config is not provided
        if camera_config is None:
            fx, fy = tfm_data["fl_x"], tfm_data["fl_y"]
            cx, cy = tfm_data["cx"], tfm_data["cy"]
            k1, k2 = tfm_data["k1"], tfm_data["k2"]
            p1, p2 = tfm_data["p1"], tfm_data["p2"]

            camera_config = {
                "model": tfm_data["camera_model"],
                "height": tfm_data["h"],
                "width": tfm_data["w"],
                "intrinsics_matrix": [
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0]
                ],
                "distortion_coefficients": [k1, k2, p1, p2]
            }
            
        # Compute the transform using aruco markers
        Psfm, Parc = extract_positions(sfm_path, extractor_config, camera_config)
        cs, Rs, ts = ch.compute_ransac_transform(Psfm, Parc)

        # Generate the sparse point cloud and transform files
        for frame in tfm_data["frames"]:
            Tc2s = np.array(frame["transform_matrix"])

            Tc2w = np.eye(4)
            Tc2w[:3, :3], Tc2w[:3, 3] = Rs @ Tc2s[:3, :3], cs * Rs @ Tc2s[:3, 3] + ts

            frame["transform_matrix"] = Tc2w.tolist()

        sparse_points = np.asarray(sparse_pcloud.points)
        for idx, point in enumerate(sparse_points):
            sparse_points[idx, :] = cs * Rs @ point + ts

        sparse_pcloud.points = o3d.utility.Vector3dVector(sparse_points)

        # Save the updated files
        with open(tfm_path, "w", encoding="utf8") as f:
            json.dump(tfm_data, f, indent=4)

        o3d.io.write_point_cloud(spc_path.as_posix(), sparse_pcloud)

    # Make the images written by Nerfstudio visible next to transforms.json.
    if use_images:
        scene_file_path = workspace_path / scene_file_name
        existing_image_path = scene_file_path / "sfm/images"
        desired_image_path = scene_file_path / "images"
        logger.info(f"Creating symlink from {existing_image_path} to {desired_image_path}")
        if desired_image_path.exists() or desired_image_path.is_symlink():
            if not desired_image_path.is_symlink() or desired_image_path.resolve() != existing_image_path.resolve():
                raise FileExistsError(
                    f"Cannot create image link at {desired_image_path}: it already exists and has a different target."
                )
        else:
            desired_image_path.symlink_to(existing_image_path.resolve(), target_is_directory=True)

    # Run the gsplat generation
    command = [
        "ns-train",
        "splatfacto",
        "--pipeline.datamanager.cache-images", "disk",
        "--data", scene_file_name,
        "--viewer.quit-on-train-completion", "True",
        "--output-dir", 'outputs',
        "--pipeline.model.camera-optimizer.mode", "SO3xR3",
        
        "--max-num-iterations", "60000" ,
        "--pipeline.model.densify-grad-thresh", "0.00015" ,
        "--pipeline.model.stop-split-at", "45000" ,
        # "--pipeline.model.cull-alpha-thresh", "0.002" , causes bad letters
        "--pipeline.model.use-scale-regularization", "True",
        # "--pipeline.model.max-gs-num", "4000000",
        "nerfstudio-data",
        "--orientation-method", "none",
        "--center-method", "none",
        "--auto-scale-poses", "False",

    ]

    def run_command_live_output(cmd):
        # Use text=True for automatic decoding of bytes to strings
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=workspace_path.as_posix())
        
        # Read the output line by line as it is produced
        for line in process.stdout:
            print(line, end='', flush=True)  # Print immediately to the console

        process.stdout.close()
        process.wait()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)
        
        return process.returncode
    
    # Run the command
    result = run_command_live_output(command)

    # Check the result
    if result == 0:
        print("Command succeeded.")
        # print(result.stdout)  # Output of the command
    else:
        print("Command failed.")
        # print(result.stdout)  # Output of the command
        # print(result.stderr)  # Error output


def extract_frames(video_path: Union[str, Path], rgbs_path: Path,
                   extractor_config: Dict[str, Union[int, float]]) -> List[np.ndarray]:
    """
    Extracts frame data from video into a folder of images.
    
    Args:
        video_path: Path to the video file
        rgbs_path: Path to save extracted images
        extractor_config: Configuration for frame extraction
    
    Returns:
        List of extracted frames
    """
    # Unpack the extractor configs
    Nimg = extractor_config["num_images"]
    Narc = extractor_config["num_marked"]
    mkr_id = extractor_config["marker_id"]

    # Initialize the aruco detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    # Open the video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Error: Cannot open the video file.")
    
    # Survey frames for aruco markers
    Ntot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    Tarc, Temp = [], []
    for _ in range(Ntot):
        ret, frame = cap.read()
        if not ret:
            break

        # Check if the frame has an aruco marker
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, ids, _ = detector.detectMarkers(gray)

        # Bin the frame by the marker detection
        if ids is not None and len(ids) == 1 and ids[0] == mkr_id:
            Tarc.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        else:
            Temp.append(cap.get(cv2.CAP_PROP_POS_MSEC))

    # Check if enough aruco markers were found
    if len(Tarc) < Narc:
        Tout = Tarc + ch.distribute_values(Temp, Nimg - len(Tarc))
        print(f"Warning: Only {len(Tarc)} aruco markers found. Using {Narc - len(Tarc)} empty frames to fill the gap.")
    else:
        Tout = ch.distribute_values(Tarc, Narc) + ch.distribute_values(Temp, Nimg - Narc)
    
    Tout.sort()

    # Extract the selected frames
    frames = []
    for idx, tout in enumerate(Tout):
        cap.set(cv2.CAP_PROP_POS_MSEC, tout)
        ret, frame = cap.read()
        if not ret:
            break

        # Save the image
        rgb_path = rgbs_path / f"frame_{idx+1:05d}.png"
        cv2.imwrite(str(rgb_path), frame)
        frames.append(frame)

    # Release the video capture object
    cap.release()
    
    return frames


def extract_positions(sfm_path: Path,
                      extractor_config: Dict[str, Union[int, float]],
                      camera_config: Optional[Dict[str, Union[int, float]]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract positions from SfM results using Aruco markers.
    
    Args:
        sfm_path: Path to the SfM output directory
        extractor_config: Configuration for marker extraction
        camera_config: Camera configuration
    
    Returns:
        Tuple of (Psfm, Parc) where:
            Psfm: 3xN array of SfM camera positions
            Parc: 3xN array of Aruco marker positions
    """
    # Unpack the extractor configs
    Narc = extractor_config["num_marked"]
    marker_length = extractor_config["marker_length"]
    marker_id = extractor_config["marker_id"]

    # A rig stores camera calibration on each frame; a regular capture uses the
    # supplied shared calibration.
    if camera_config is not None:
        camera_matrix = np.array(camera_config["intrinsics_matrix"])
        dist_coeffs = np.array(camera_config["distortion_coefficients"])

    # Initialize the aruco detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    marker_points = np.array([
        [-marker_length / 2,  marker_length / 2, 0],
        [ marker_length / 2,  marker_length / 2, 0],
        [ marker_length / 2, -marker_length / 2, 0],
        [-marker_length / 2, -marker_length / 2, 0]
    ])
    
    # Open the transforms.json file
    with open(sfm_path / "transforms.json", "r") as f:
        transforms = json.load(f)
    frames = transforms["frames"]

    TTarc, TTsfm = [], []
    for frame in tqdm(frames):
        if camera_config is None:
            camera_matrix = np.array([
                [frame["fl_x"], 0.0, frame["cx"]],
                [0.0, frame["fl_y"], frame["cy"]],
                [0.0, 0.0, 1.0],
            ])
            dist_coeffs = np.array([
                frame.get("k1", 0.0), frame.get("k2", 0.0),
                frame.get("p1", 0.0), frame.get("p2", 0.0),
            ])

        # Open the image file
        # image_path = sfm_path.parent / frame["file_path"]
        image_path = sfm_path / frame["file_path"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Error: Cannot open the image file {image_path}")
        
        # Detect the aruco marker in the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None and len(ids) == 1 and ids[0] == marker_id:            
            # Compute the Aruco transform
            ret, rvec, tvec = cv2.solvePnP(
                marker_points, corners[0], camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            
            # Compute the transforms
            if ret:
                Tw2c_arc = np.eye(4)
                Tw2c_arc[:3, :3], Tw2c_arc[:3, 3] = cv2.Rodrigues(rvec)[0], tvec.flatten()  # world to camera

                # Compute the camera to world transforms
                Tarc = np.linalg.inv(Tw2c_arc)
                Tsfm = np.array(frame["transform_matrix"])

                TTarc.append(Tarc)
                TTsfm.append(Tsfm)
    
    # Check if the number of transforms match our expectations
    if Narc != -1:
        if len(TTarc) != Narc:
            raise ValueError("Error: Mismatched number of aruco and sfm transforms.")
    else:
        assert len(TTarc) > 0

    # Extract the positions
    Parc = np.array([T[:3, 3] for T in TTarc]).T
    Psfm = np.array([T[:3, 3] for T in TTsfm]).T

    return Psfm, Parc
