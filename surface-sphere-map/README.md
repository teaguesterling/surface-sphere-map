surface-sphere-map
==================

Running python surface_mapper.py

usage: surface_mapper.py [-h] [--voxel-resolution VOXEL_RESOLUTION]
                         [--voxel-padding VOXEL_PADDING]
                         [--contour-faces CONTOUR_FACES]
                         [--contour-scale CONTOUR_SCALE]
                         [--normalize-contour-distance NORMALIZE_CONTOUR_DISTANCE]
                         [--timestep TIMESTEP] [--cvf-scale CVF_SCALE]
                         [--gvf-scale GVF_SCALE] [--push-scale PUSH_SCALE]
                         [--snap-scale SNAP_SCALE]
                         [--internal-scale INTERNAL_SCALE]
                         [--internal-tension INTERNAL_TENSION]
                         [--internal-stiffness INTERNAL_STIFFNESS]
                         [--gvf-smoothness GVF_SMOOTHNESS]
                         [--gvf-timestep GVF_TIMESTEP]
                         [--gvf-max-steps GVF_MAX_STEPS]
                         [--gvf-normalize-distance GVF_NORMALIZE_DISTANCE]
                         [--cvf-normalize-distance CVF_NORMALIZE_DISTANCE]
                         [--convergence-threshold CONVERGENCE_THRESHOLD]
                         [--converge-on-snapping CONVERGE_ON_SNAPPING]
                         [--max-iterations MAX_ITERATIONS]
                         [--visualize {convergence,mapping,none}]
                         [--visualize-file VISUALIZE_FILE]
                         [--visualize-every VISUALIZE_EVERY]
                         [--show-accuracy SHOW_ACCURACY]
                         [surface_file] [mapping_file]

positional arguments:
  surface_file          Vet (MSRoll) file to map [Default: -]
  mapping_file          Destination to write sphere mapping [Default: -]

optional arguments:
  -h, --help            show this help message and exit
  --voxel-resolution VOXEL_RESOLUTION
                        Resolution at which to voxelize surface (in Angstroms)
                        [Default: 0.75]
  --voxel-padding VOXEL_PADDING
                        Padding to add to voxelized image [Default: 5.0]
  --contour-faces CONTOUR_FACES
                        Number of faces to generate in contour [Default: >=
                        source mesh]
  --contour-scale CONTOUR_SCALE
                        Scale factor for contour [Default: 1.0]
  --normalize-contour-distance NORMALIZE_CONTOUR_DISTANCE
                        Normalize contour distance to protein surface
                        [Default: False] (Warning: May require signifigant
                        padding!)
  --timestep TIMESTEP   Global system timestep [Default: 0.15]
  --cvf-scale CVF_SCALE
                        Curvature Vector Flow scale factor [Default: 0.75]
  --gvf-scale GVF_SCALE
                        Gradient Vector Flow scale factor [Default: 0.5]
  --push-scale PUSH_SCALE
                        Boundary push scale factor [Default: 1]
  --snap-scale SNAP_SCALE
                        Surface snapping scale factor [Default: 1]
  --internal-scale INTERNAL_SCALE
                        Global internal energy scale factor [Default: 1]
  --internal-tension INTERNAL_TENSION
                        Internal tension (elasticity) energy [Default: 0.5]
  --internal-stiffness INTERNAL_STIFFNESS
                        Internal stiffness (rigidity) energy [Default: 0.35]
  --gvf-smoothness GVF_SMOOTHNESS
                        Gradient vector flow smoothness parameter [Default:
                        0.1]
  --gvf-timestep GVF_TIMESTEP
                        Gradient vector flow generation timestep [Default:
                        0.15]
  --gvf-max-steps GVF_MAX_STEPS
                        Gradient vector flow iterations [Default: Guessed]
  --gvf-normalize-distance GVF_NORMALIZE_DISTANCE
                        Normalize GVF after Euclidian distance from surface
                        term [Default: 10]
  --cvf-normalize-distance CVF_NORMALIZE_DISTANCE
                        Normalize CVF after Euclidian distance from surface
                        term [Default: 15]
  --convergence-threshold CONVERGENCE_THRESHOLD
                        Average movement threshold for termination [Default:
                        0.015]
  --converge-on-snapping CONVERGE_ON_SNAPPING
                        Only consider snapping in convergence [Default: False]
  --max-iterations MAX_ITERATIONS
                        Maxiumum number of iterations to run before forced
                        termination [Default: 100]
  --visualize {convergence,mapping,none}
                        Display visualization (Requires MayaVI) [Default:
                        none]
  --visualize-file VISUALIZE_FILE
                        Display visualization (Requires MayaVI) [Default:
                        none]
  --visualize-every VISUALIZE_EVERY
                        Interrupt iterations to visualize every N steps
                        [Default: None]
  --show-accuracy SHOW_ACCURACY
                        Compute and display mapping quality information
                        [Default: True]

