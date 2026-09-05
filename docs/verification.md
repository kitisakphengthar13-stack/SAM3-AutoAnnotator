# Verification

This document separates repeatable software checks from hardware-dependent
SAM3 inference. Run commands from the repository root.

## Environment inventory

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pip list
```

`pip` in these commands inspects third-party dependencies only. There is no
editable/project installation to verify.

## Static import and bytecode check

```powershell
.\.venv\Scripts\python.exe -m compileall -q main.py sam3_auto_annotator tests
```

Success means the command exits with code `0`; it does not prove GUI behavior or
model inference.

## Automated tests

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
Remove-Item Env:QT_QPA_PLATFORM -ErrorAction SilentlyContinue
```

The suite covers core annotation rules, project serialization/export
preparation, annotation services, SAM3 result mapping with fakes, predictor
cache/precision configuration, Qt fields/models, canvas segmentation behavior,
batch target selection, and YOLO import. Offscreen Qt tests do not require a
GPU.

Record the date, Python/PySide6 versions, test count, and exact result in the
handoff after running this command. Do not treat the command's presence in this
document as a passing result.

### Latest repeatable baseline

Verified on 2026-09-05 with Python 3.14.7, PySide6 6.11.2, Pillow 12.3.0,
and Ultralytics 8.4.133:

- `pip check`: no broken requirements;
- `compileall`: passed;
- unit/integration/offscreen Qt suite: **112 tests passed**;
- offscreen application composition: opened and closed successfully;
- layout captures: checked at 960×620 and 1360×840 with no inspector
  horizontal overflow and stable splitter proportions;
- real checkpoint inference: not run under the battery-power constraint below.

## Offscreen application smoke test

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\python.exe -c "from sam3_auto_annotator.application import create_application; app, window = create_application([]); window.show(); app.processEvents(); window.close(); print('GUI startup OK')"
Remove-Item Env:QT_QPA_PLATFORM -ErrorAction SilentlyContinue
```

This verifies composition and first render without opening a visible window. It
does not validate dialogs, mouse interaction, output content, or SAM3.

## Manual GUI acceptance check

Run on a normal desktop session:

```powershell
.\.venv\Scripts\python.exe main.py
```

Check the complete product story:

1. Open `images_test` and confirm all three car images appear.
2. Search/filter the Dataset list and navigate with toolbar and shortcuts.
3. Configure `models\sam3.1_multiplex.pt`, prompt `car`, confidence, and
   precision.
4. Draw, resize, reclassify, delete, reset, and review annotations.
5. Save the project; close and reopen `annotation_state.json`.
6. Import a small YOLO detection-label directory and verify box placement.
7. Export and inspect the CSV, detection labels, segmentation labels, skipped
   report when applicable, summary, and preview.
8. Resize the main window and all splitter regions; restart and confirm geometry
   restoration.
9. Start a pending-image batch, request cancellation, and verify the current
   item finishes cleanly without overwriting edited/reviewed work.

## Real SAM3 GPU check — pending in the latest handoff

The latest work session did **not** run real GPU inference. The development
laptop was on battery power, and loading/running the approximately 3.5 GB SAM3
checkpoint would violate the requested battery constraint. No passing GPU claim
is made here.

When the laptop is connected to external power, first confirm that CUDA is
available:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print({'cuda_available': torch.cuda.is_available(), 'device_count': torch.cuda.device_count(), 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})"
```

The preferred end-to-end check is then the visible GUI flow above: open
`images_test`, select `models\sam3.1_multiplex.pt`, enter `car`, run the
current image, use **Run Pending**, review, save, and export.

For a focused service-level diagnostic before the visible workflow, run:

```powershell
.\.venv\Scripts\python.exe -c "from pathlib import Path; from sam3_auto_annotator.services.prediction_service import PredictionService; r = PredictionService().predict_image(image_path=Path('images_test/car_1.jpg'), model_path=Path('models/sam3.1_multiplex.pt'), prompts=['car'], confidence=0.5, half=True); print({'image': str(r.image_path), 'annotations': len(r.annotations), 'size': (r.width, r.height), 'reused_predictor': r.reused_predictor})"
```

Before considering the GPU path verified, confirm all of the following:

- the process selects the intended CUDA device and does not fall back silently;
- first inference completes without out-of-memory or precision errors;
- a second image reuses the predictor for identical settings;
- results appear on the GUI thread without freezing or unsafe thread shutdown;
- saved state reopens with the same boxes/classes/polygon validity;
- exported detection/segmentation counts agree with the reviewed project state.

If inference fails, record the exception, CUDA/driver versions, free VRAM, model
path, precision, confidence, and prompt. A CPU or fake-predictor pass must not be
reported as a real GPU pass.
