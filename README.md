# Autonomous Scrapyard Monitoring Drone --- AI Prototype

Computer-vision prototype for outdoor scrapyard monitoring, initially
targeting **fire and smoke detection**.

> **Status:** A YOLO11n fire/smoke model has been trained as an initial
> prototype. A VS Code image/webcam testing workflow has been prepared.
> Field performance and drone integration are not yet validated.

## Project overview

Planned monitoring targets: - **Fire** - **Smoke** - **People**
*(planned; not included in the current trained model)*

## Current system

  ---------------------------------------------------------------------------------------------------------------------
  Component                           Status / selection
  ----------------------------------- ---------------------------------------------------------------------------------
  Drone                               Custom quadcopter

  Flight controller                   Pixhawk 6X

  Environment                         Outdoor scrapyard

  Planned flight height               5 m AGL (project target, not a validated safe altitude)

  Dataset                             [Smoke-Fire Detection YOLO ---
                                      Kaggle](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo)

  Model                               Ultralytics YOLO11n, fire/smoke

  Training                            Google Colab, GPU when available

  Local testing                       VS Code, Python

  Vision                              OpenCV

  ML framework                        PyTorch / Ultralytics

  Camera and onboard computer         Selection/integration pending
  ---------------------------------------------------------------------------------------------------------------------

## Work completed

-   Selected a Kaggle YOLO-format fire/smoke dataset for initial
    development.
-   Prepared a Colab dataset/training workflow.
-   Trained a YOLO11n fire/smoke detector.
-   Reviewed training curves, metrics, F1-confidence curve, and
    confusion matrices.
-   Prepared local image and webcam inference workflow in VS Code.
-   Considered separate COCO-pretrained YOLO person detection as an
    optional prototype addition.

**Dataset caution:** Check the downloaded dataset YAML for class names,
IDs, and split paths. Do not assume class ordering or remap annotations
without verifying the original labels.

## Preliminary training results

The provided plots show results through **epoch 10**:

  Metric                            Approximate plotted value
  ------------------------------- ---------------------------
  Precision (B)                                         0.714
  Recall (B)                                            0.593
  mAP@50 (B)                                            0.673
  mAP@50--95 (B)                                        0.367
  Best overall F1 shown                                  0.65
  Confidence at plotted F1 peak                         0.264

These are preliminary plot readings, not final held-out test or field
results. The F1 peak threshold is a candidate for further validation,
not a deployment recommendation. The confusion matrix shows substantial
background-associated errors; inspect false positives/misses and verify
the evaluation setup.

## Suggested repository structure

``` text
scrapyard-drone-ai/
├── models/
│   └── fire_smoke_best.pt
├── images/
│   └── test.jpg
├── outputs/
├── detect.py
├── requirements.txt
└── README.md
```

-   `models/fire_smoke_best.pt`: trained checkpoint copied from the
    training output.
-   `images/test.jpg`: sample input image.
-   `outputs/`: annotated predictions.
-   `detect.py`: local image/webcam inference script.

## Setup (Windows / VS Code)

From the project root:

``` bash
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install ultralytics opencv-python
```

If `opencv-python-headless` is installed in the same environment, remove
it to avoid GUI package conflicts:

``` bash
pip uninstall opencv-python-headless -y
pip install opencv-python
```

### `requirements.txt`

``` text
ultralytics
opencv-python
```

For NVIDIA GPU inference, install a PyTorch build compatible with your
driver/system using the official [PyTorch installation
guide](https://pytorch.org/get-started/locally/). CPU inference is also
possible.

## Run inference

### Image

``` bash
python detect.py --mode image
```

Specify an image:

``` bash
python detect.py --mode image --image images/test.jpg
```

Optional separate person detector:

``` bash
python detect.py --mode image --person
```

Annotated output is expected under `outputs/`.

### Webcam

``` bash
python detect.py --mode webcam
```

With optional person detection:

``` bash
python detect.py --mode webcam --person
```

Try another camera index if needed:

``` bash
python detect.py --mode webcam --camera 1
```

Press **Q** to exit.

The displayed processing FPS is approximate loop throughput, not full
end-to-end latency or a flight-readiness metric. Running a second
detector can reduce throughput.

## Next steps

1.  Confirm training completion and document the
    checkpoint/configuration.
2.  Evaluate on a held-out **test** split if available; otherwise label
    results as validation.
3.  Record per-class fire and smoke precision, recall, mAP@50, and
    mAP@50--95.
4.  Review false positives/negatives on scrap, rust, sunlight, dust,
    clouds, and exhaust.
5.  Test on representative images and videos, not only training samples.
6.  Measure sustained throughput and end-to-end latency with documented
    hardware/settings.
7.  Collect and annotate relevant imagery from the intended
    camera/viewpoint, with site permission and safety precautions.
8.  Validate person detection and compare separate versus unified
    models.
9.  Choose camera/onboard computer and benchmark the complete pipeline
    on target hardware.
10. Consider flight-stack integration only after perception validation
    and safety review.

## Limitations and safety

-   Current trained model: **fire and smoke only**; person detection is
    not included.
-   Dataset performance may not transfer to scrapyard scenes or aerial
    viewpoints.
-   Performance at the intended 5 m height is unverified.
-   Webcam/offline-video success does not establish airborne
    reliability.
-   Do not use this prototype as the sole fire alarm, safety system, or
    flight-control trigger.
-   Keep flight stabilization and failsafes independent from this
    perception prototype; test under controlled conditions with human
    oversight.

## References

-   Dataset: [Smoke-Fire Detection YOLO
    (Kaggle)](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo)
-   [Ultralytics documentation](https://docs.ultralytics.com/)
-   [PyTorch installation
    guide](https://pytorch.org/get-started/locally/)

------------------------------------------------------------------------

**Project:** Autonomous Scrapyard Monitoring Drone\
**AI prototype:** Fire and smoke detection\
**Workflow:** Google Colab → VS Code → future target-hardware validation
