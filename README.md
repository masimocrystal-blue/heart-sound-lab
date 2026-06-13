# Heart Sound Lab

Procedural heart sound synthesis experiments in Python.

This repository contains scripts for generating synthetic cardiac auscultation sounds,
including normal heart sounds, respiratory sinus arrhythmia, split S2,
different auscultation sites, and sound-design-oriented variations.

## Project Modes

### 1. Auscultation-oriented synthesis

Sounds designed to resemble body-surface stethoscope recordings.

### 2. Music-oriented synthesis

Sounds designed as cardiac-inspired audio material for electronic music,
sound design, techno, electronica, and experimental audio.

## Disclaimer

The generated sounds are synthetic.
They are not clinical recordings, medical devices, diagnostic tools, or validated teaching references.

Do not use these sounds to diagnose, train diagnostic systems, evaluate patients,
or make medical decisions.

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

Run an example script:

```bash
python scripts/claude_apex_realistic.py
```

Generated audio files should be written to `outputs/`.

## License

MIT License.
