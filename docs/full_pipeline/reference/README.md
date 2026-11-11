# MARVEL companion reference

This folder contains the full source printed in the numbered guides one directory above. Read `../00_START_HERE.md` first.

Run commands from this directory. On Windows, select your project environment and set:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m unittest discover -s tests -v
```

Create local `data`, `models` and `reports` folders for your actual assets and outputs. No model weights, recordings, secrets or validated hardware environment are included.

The root application's existing scaffold is separate. Integrate reviewed components with teammates' current code rather than overwriting it blindly.
