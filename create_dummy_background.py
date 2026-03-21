#!/usr/bin/env python3
import numpy as np
from pathlib import Path

def main():
    bg_dir = Path('/data/background')
    bg_dir.mkdir(parents=True, exist_ok=True)
    
    npz_path = bg_dir / 'background_voxel_map.npz'
    
    # 全ての点群を「前景」として扱うための空の背景モデルを作成する
    voxel_indices = np.empty((0, 3), dtype=np.int64)
    voxel_size = np.float32(0.1)
    roi_min = np.array([0.0, -2.0, 0.1], dtype=np.float32)
    roi_max = np.array([5.0, 2.0, 2.0], dtype=np.float32)
    
    np.savez(
        npz_path,
        voxel_indices=voxel_indices,
        voxel_size=voxel_size,
        roi_min=roi_min,
        roi_max=roi_max
    )
    print(f"Dummy background model created at {npz_path}")

if __name__ == '__main__':
    main()
