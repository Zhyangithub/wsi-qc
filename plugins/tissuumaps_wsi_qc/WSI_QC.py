"""Server side of the WSI QC TissUUmaps plugin.

This module exposes a single API endpoint ``run_pipeline`` which executes the
quality‑control pipeline on the server and writes TissUUmaps‑ready overlay CSVs
to the output directory.  The JavaScript front end triggers this endpoint and
then adds the generated overlays as layers.
"""

import logging
import os
import subprocess

from flask import abort, make_response


class Plugin:
    """TissUUmaps server plugin for WSI QC."""

    def __init__(self, app):
        self.app = app

    def run_pipeline(self, json_param):
        """Run the QC pipeline with parameters provided by the UI.

        Parameters
        ----------
        json_param: dict
            Parameters forwarded from the JavaScript side.  Expected keys are
            ``wsi`` (path to input slide), ``out`` (output directory),
            ``ref`` (optional stain reference JSON), ``tile`` (tile size), and
            ``qc_thresh``.  Only ``wsi`` and ``out`` are mandatory.

        Returns
        -------
        flask.Response
            Simple success message if the pipeline finished without errors.
        """

        if not json_param:
            logging.error("No arguments supplied to run_pipeline")
            abort(500)

        wsi = json_param.get("wsi")
        out_dir = json_param.get("out")
        ref = json_param.get("ref")
        tile = str(json_param.get("tile", 512))
        qc_thresh = str(json_param.get("qc_thresh", 0.80))

        if not wsi or not out_dir:
            logging.error("Missing required arguments: wsi=%s, out_dir=%s", wsi, out_dir)
            abort(500)

        tiles_dir = os.path.join(out_dir, "tiles")
        qc_csv = os.path.join(out_dir, "qc.csv")

        try:
            subprocess.run([
                "python",
                "scripts/tile_wsi.py",
                "--wsi",
                wsi,
                "--out",
                out_dir,
                "--tile",
                tile,
            ], check=True)

            subprocess.run([
                "python",
                "scripts/qc_score_tiles.py",
                "--tiles",
                tiles_dir,
                "--out",
                qc_csv,
            ], check=True)

            for metric, name in [
                ("qc_score", "qc"),
                ("stripe_score", "stripe"),
                ("bubble_score", "bubble"),
                ("fold_score", "fold"),
            ]:
                subprocess.run([
                    "python",
                    "scripts/export_heatmap_to_tissuumaps.py",
                    "--tile_csv",
                    qc_csv,
                    "--metric",
                    metric,
                    "--out",
                    os.path.join(out_dir, f"overlay_{name}.csv"),
                ], check=True)

        except subprocess.CalledProcessError:
            logging.exception("QC pipeline failed")
            abort(500)

        return make_response("QC pipeline completed")
