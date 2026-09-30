# Fly Me to the Flies

PC prototype for a fruit-fly catching game.

## Current prototype

- Anatomically detailed Drosophila body from flybody
- MuJoCo physics simulation
- Enclosed room with floor, walls, and ceiling
- Fly flight driven by flybody's existing wing-beat/flight machinery
- Keyboard-controlled 3D swatter
- Contact-based catch detection
- Reset after a catch

The flybody model is installed as a Python dependency instead of being copied into this repository.

## Requirements

- Python 3.10+
- A desktop OS with MuJoCo rendering support
- Python packages listed in `requirements.txt`

## Run

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

If PowerShell blocks script activation, run:

```powershell
.\\.venv\\Scripts\\python.exe main.py
```

## Controls

- W / S: move swatter forward / backward
- A / D: move swatter left / right
- Q / E: move swatter down / up
- SPACE: reset swatter to its starting position
- R: reset the fly after it has been caught
- ESC: quit

The first version deliberately keeps the player interaction simple. The next stage can replace keyboard control with mouse aiming and add more realistic fly sensory/autonomous behavior without changing the underlying flybody model.
