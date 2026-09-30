# Fly Me to the Flies

PC prototype for a fruit-fly catching game.

## Current prototype

- Anatomically detailed Drosophila body from FlyBody
- Enclosed room with floor, walls, and ceiling
- Mouse-look first-person camera
- Left-click swatter attack
- Measured FlyBody flight trajectories replayed at 60 FPS
- Fly behavior states: flying, landing, grounded, and takeoff
- No real-time whole-body flight physics in the game loop

The game separates biological motion data from interactive rendering. The official FlyBody flight-imitation dataset contains measured trajectories including saccades, evasion maneuvers, altitude changes, sideways/backward flight, and hovering. The full FlyBody physics/controller environment remains useful for research, but it is too expensive to run as the gameplay clock.

## Requirements

- Python 3.10+
- A desktop OS with MuJoCo rendering support
- Python packages listed in `requirements.txt`
- The FlyBody flight-imitation HDF5 dataset

## Run

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python download_motion_data.py
python main.py
```

If PowerShell blocks script activation, run:

```powershell
.\\.venv\\Scripts\\python.exe download_motion_data.py
.\\.venv\\Scripts\\python.exe main.py
```

### Motion dataset

`download_motion_data.py` downloads the official FlyBody supporting archive from Janelia Figshare and extracts only `data/flight-dataset_saccade-evasion_augmented.hdf5`.

The Figshare archive is about 4 GB compressed, even though the flight HDF5 needed by this prototype is much smaller. The upstream dataset is GPL 3.0+ and should be kept separate from the project's own code license.

## Controls

- Mouse: look around
- Left mouse button: swat
- F1: release/grab the mouse
- ESC: quit

There is intentionally no WASD/QE movement. The player remains stationary and catches the fly by aiming the swatter.

## Why the game does not run FlyBody physics every frame

The published FlyBody flight model is an anatomically detailed 102-DoF whole-body MuJoCo simulation. Its research flight controller operates at very small physics/control timesteps, so using it as the 60 FPS gameplay clock causes severe CPU load.

Instead, this prototype uses measured flight-imitation trajectories as a kinematic replay. MuJoCo is still used to render the anatomical fly and room, but the game loop does not call `env.step()`.

This also removes the previous failure mode where a terminal physics timestep caused the fly to be reset to its initial position.

## Research references

- FlyBody: https://github.com/TuragaLab/flybody
- FlyBody supporting dataset: https://janelia.figshare.com/articles/dataset/MuJoCo_fruit_fly_body_model_datasets_supporting_Whole-body_simulation_of_realistic_fruit_fly_locomotion_with_deep_reinforcement_learning_/25309105
- Vaxenburg et al., *Whole-body physics simulation of fruit fly locomotion*, Nature (2025): https://doi.org/10.1038/s41586-025-09029-4

The measured replay is a visualization/gameplay layer, not a claim that the game is running the trained FlyBody neural controller or a biologically complete connectome.
