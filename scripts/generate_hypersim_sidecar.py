import os
import sys
import h5py
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from dotenv import load_dotenv

from open_vocab_mot import HYPERSIM_DATASET_PATH

load_dotenv(verbose=True, override=True)

ML_HYPERSIM_PATH = Path(os.getenv("ML_HYPERSIM_PATH", "/z/dat/ml-hypersim"))
if not ML_HYPERSIM_PATH.exists():
    print(f"Please clone the hypersim repo and set ML_HYPERSIM_PATH to the cloned path")
    exit(0)

def get_asset_id_mapping(scene_name: str) -> dict[int, str]:
    mapping = {}
    mesh_dir = ML_HYPERSIM_PATH / "evermotion_dataset" / "scenes" / scene_name / "_detail" / "mesh"
    metadata_objects_path = mesh_dir / "metadata_objects.csv"
    sii_path = mesh_dir / "mesh_objects_sii.hdf5"
    
    if not metadata_objects_path.exists() or not sii_path.exists():
        return mapping
        
    import csv
    import re
    try:
        object_id_to_name = {}
        with open(metadata_objects_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            for i, row in enumerate(reader):
                if row:
                    object_id_to_name[i] = row[0]
                    
        with h5py.File(sii_path, "r") as f:
            # Flatten the (N, 1) array so we get scalars when iterating
            sii_data = f["dataset"][:].flatten()
            
        sem_inst_to_base_names = {}
        for obj_id, sem_inst_id in enumerate(sii_data):
            sem_inst_id = int(sem_inst_id)
            if sem_inst_id == -1: continue
            obj_name = object_id_to_name.get(obj_id, "")
            if not obj_name: continue
            
            # Extract base name by removing _obj_...
            base_name = re.sub(r'_obj_?\d*$', '', obj_name).strip('_')
            # Also remove trailing digits (e.g. chair1 -> chair, chair2 -> chair)
            base_name = re.sub(r'\d+$', '', base_name).strip('_')
            
            if sem_inst_id not in sem_inst_to_base_names:
                sem_inst_to_base_names[sem_inst_id] = set()
            sem_inst_to_base_names[sem_inst_id].add(base_name)
            
        mapping = {k: "+".join(sorted(list(v))) for k, v in sem_inst_to_base_names.items()}
    except Exception as e:
        print(f"Error processing asset mapping for {scene_name}: {e}")
        
    return mapping

def process_scene(scene_dir: Path):
    print(f"Started processing {scene_dir}")
    scene_name = scene_dir.name
    scene_data = {"objects": {}}
    
    asset_id_mapping = get_asset_id_mapping(scene_name)
    
    images_dir = scene_dir / "images"
    if not images_dir.exists(): 
        return scene_name, scene_data
    
    cam_dirs = [d for d in images_dir.iterdir() if d.is_dir() and "geometry_hdf5" in d.name]
    
    for cam_dir in cam_dirs:
        # e.g., scene_cam_00_geometry_hdf5 -> cam_00
        parts = cam_dir.name.split("_geometry")
        if len(parts) < 1: continue
        cam_name = parts[0]
        if "scene_" in cam_name:
            cam_name = cam_name.split("scene_")[1]
            
        frames = sorted(list(cam_dir.glob("*.semantic_instance.hdf5")))
        
        for frame in frames:
            frame_name_parts = frame.name.split(".")
            frame_name = f"{frame_name_parts[0]}.{frame_name_parts[1]}"
            
            sem_inst_path = frame
            sem_path = frame.parent / f"{frame_name}.semantic.hdf5"
            
            if not sem_inst_path.exists() or not sem_path.exists():
                continue
                
            try:
                with h5py.File(sem_inst_path, "r") as f_inst, h5py.File(sem_path, "r") as f_sem:
                    inst_data = f_inst["dataset"][:]
                    sem_data = f_sem["dataset"][:]
                    
                    unique_insts = np.unique(inst_data)
                    for inst in unique_insts:
                        if inst == -1: continue
                        
                        mask = inst_data == inst
                        if not np.any(mask): continue
                        
                        sem_label = int(sem_data[mask][0])
                        obj_id_str = str(int(inst))
                        
                        if obj_id_str not in scene_data["objects"]:
                            asset_id = asset_id_mapping.get(int(inst), obj_id_str)
                            scene_data["objects"][obj_id_str] = {
                                "semantic_label": sem_label,
                                "asset_id": asset_id,
                                "cameras": {}
                            }
                            
                        if cam_name not in scene_data["objects"][obj_id_str]["cameras"]:
                            scene_data["objects"][obj_id_str]["cameras"][cam_name] = []
                            
                        scene_data["objects"][obj_id_str]["cameras"][cam_name].append(frame_name)
            except Exception as e:
                pass
                
    return scene_name, scene_data

def main():
    try:
        from open_vocab_mot.definitions import HYPERSIM_DATASET_PATH, HYPERSIM_DATASET_SIDECAR_PATH
    except ImportError:
        print("Please run this script from the project root or ensure open_vocab_mot is installed.")
        sys.exit(1)
        
    root_dir = HYPERSIM_DATASET_PATH
    
    if not root_dir.exists():
        print(f"Dataset frames directory not found at {root_dir}")
        sys.exit(1)
        
    scene_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir()])
    print(f"Found {len(scene_dirs)} scenes.")
    
    final_data = {"scenes": {}}
    
    # We use multiple processes because HDF5 reads are CPU/IO bound
    with ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as executor:
        futures = {executor.submit(process_scene, d): d for d in scene_dirs}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing scenes"):
            scene_name, scene_data = future.result()
            print(f"Finished processing {scene_name} which had {len(scene_data['objects'])} objects.")
            if len(scene_data["objects"]) > 0:
                final_data["scenes"][scene_name] = scene_data
                
    HYPERSIM_DATASET_SIDECAR_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HYPERSIM_DATASET_SIDECAR_PATH, "w") as f:
        json.dump(final_data, f, indent=2)
        
    print(f"Successfully generated sidecar at {HYPERSIM_DATASET_SIDECAR_PATH}")

if __name__ == "__main__":
    main()
