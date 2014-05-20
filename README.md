surface-sphere-map
==================


Teague Sterling
2014

Dependencies:
  - Python 2.7.x
  - Numpy 1.6
  - Scipy
  - MSRoll 

Preprocessing: 
    msroll -m PDBID.pdb -t PDBID.vet

Execution with default parameters: 
    python surface\_mapper.py PDBID.vet PDBID.sphmap [OPTIONS]

Command Help (python surface\_mapper.py --help):
    usage: surface\_mapper.py [-h] [--voxel-resolution VOXEL\_RESOLUTION]
                             [--voxel-padding VOXEL\_PADDING]
                             [--contour-faces CONTOUR\_FACES]
                             [--contour-scale CONTOUR\_SCALE]
                             [--normalize-contour-distance NORMALIZE\_CONTOUR\_DISTANCE]
                             [--timestep TIMESTEP] [--cvf-scale CVF\_SCALE]
                             [--gvf-scale GVF\_SCALE] [--push-scale PUSH\_SCALE]
                             [--snap-scale SNAP\_SCALE]
                             [--internal-scale INTERNAL\_SCALE]
                             [--internal-tension INTERNAL\_TENSION]
                             [--internal-stiffness INTERNAL\_STIFFNESS]
                             [--gvf-smoothness GVF\_SMOOTHNESS]
                             [--gvf-timestep GVF\_TIMESTEP]
                             [--gvf-max-steps GVF\_MAX\_STEPS]
                             [--gvf-normalize-distance GVF\_NORMALIZE\_DISTANCE]
                             [--cvf-normalize-distance CVF\_NORMALIZE\_DISTANCE]
                             [--convergence-threshold CONVERGENCE\_THRESHOLD]
                             [--converge-on-snapping CONVERGE\_ON\_SNAPPING]
                             [--max-iterations MAX\_ITERATIONS]
                             [--visualize {convergence,mapping,none}]
                             [--visualize-file VISUALIZE\_FILE]
                             [--visualize-every VISUALIZE\_EVERY]
                             [--show-accuracy SHOW\_ACCURACY]
                             [surface\_file] [mapping\_file]
    
    positional arguments:
      surface\_file          Vet (MSRoll) file to map [Default: -]
      mapping\_file          Destination to write sphere mapping [Default: -]
    
    optional arguments:
      -h, --help            show this help message and exit
      --voxel-resolution VOXEL\_RESOLUTION
                            Resolution at which to voxelize surface (in Angstroms)
                            [Default: 0.75]
      --voxel-padding VOXEL\_PADDING
                            Padding to add to voxelized image [Default: 5.0]
      --contour-faces CONTOUR\_FACES
                            Number of faces to generate in contour [Default: >=
                            source mesh]
      --contour-scale CONTOUR\_SCALE
                            Scale factor for contour [Default: 1.0]
      --normalize-contour-distance NORMALIZE\_CONTOUR\_DISTANCE
                            Normalize contour distance to protein surface
                            [Default: False] (Warning: May require signifigant
                            padding!)
      --timestep TIMESTEP   Global system timestep [Default: 0.15]
      --cvf-scale CVF\_SCALE
                            Curvature Vector Flow scale factor [Default: 0.75]
      --gvf-scale GVF\_SCALE
                            Gradient Vector Flow scale factor [Default: 0.5]
      --push-scale PUSH\_SCALE
                            Boundary push scale factor [Default: 1]
      --snap-scale SNAP\_SCALE
                            Surface snapping scale factor [Default: 1]
      --internal-scale INTERNAL\_SCALE
                            Global internal energy scale factor [Default: 1]
      --internal-tension INTERNAL\_TENSION
                            Internal tension (elasticity) energy [Default: 0.5]
      --internal-stiffness INTERNAL\_STIFFNESS
                            Internal stiffness (rigidity) energy [Default: 0.35]
      --gvf-smoothness GVF\_SMOOTHNESS
                            Gradient vector flow smoothness parameter [Default:
                            0.1]
      --gvf-timestep GVF\_TIMESTEP
                            Gradient vector flow generation timestep [Default:
                            0.15]
      --gvf-max-steps GVF\_MAX\_STEPS
                            Gradient vector flow iterations [Default: Guessed]
      --gvf-normalize-distance GVF\_NORMALIZE\_DISTANCE
                            Normalize GVF after Euclidian distance from surface
                            term [Default: 10]
      --cvf-normalize-distance CVF\_NORMALIZE\_DISTANCE
                            Normalize CVF after Euclidian distance from surface
                            term [Default: 15]
      --convergence-threshold CONVERGENCE\_THRESHOLD
                            Average movement threshold for termination [Default:
                            0.015]
      --converge-on-snapping CONVERGE\_ON\_SNAPPING
                            Only consider snapping in convergence [Default: False]
      --max-iterations MAX\_ITERATIONS
                            Maxiumum number of iterations to run before forced
                            termination [Default: 100]
      --visualize {convergence,mapping,none}
                            Display visualization (Requires MayaVI) [Default:
                            none]
      --visualize-file VISUALIZE\_FILE
                            Display visualization (Requires MayaVI) [Default:
                            none]
      --visualize-every VISUALIZE\_EVERY
                            Interrupt iterations to visualize every N steps
                            [Default: None]
      --show-accuracy SHOW\_ACCURACY
                            Compute and display mapping quality information
                            [Default: True]
    
