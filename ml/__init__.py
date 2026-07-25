"""Reusable data + training logic for the rover YOLO workflow.

The modules in this package hold every piece of dataset and model logic so the
GUI (see ``gui/``) stays thin: it calls these functions instead of embedding the
work itself. Each heavy module (``extract_frames``, ``auto_label_frames``,
``train_yolo``) also keeps a ``main()`` + argparse so long jobs can be run as
subprocesses and have their output streamed live into the GUI job log.
"""
