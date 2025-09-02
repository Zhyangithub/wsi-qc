/**
 * Front-end part of the WSI QC TissUUmaps plugin.
 *
 * The panel exposes a few parameters to run the QC pipeline and loads the
 * resulting overlay CSV files as separate layers.
 */

var WSI_QC;

WSI_QC = {
  name: "WSI QC",
  parameters: {
    _section_main: {
      label: "Pipeline",
      title: "Run QC pipeline",
      type: "section",
      collapsed: false,
    },
    _wsi: {
      label: "WSI (.svs)",
      type: "text",
      default: "",
    },
    _out: {
      label: "Output dir",
      type: "text",
      default: "",
    },
    _ref: {
      label: "Ref JSON",
      type: "text",
      default: "",
    },
    _tile: {
      label: "Tile size",
      type: "number",
      default: 512,
    },
    _qc_thresh: {
      label: "QC threshold",
      type: "number",
      default: 0.8,
      attributes: { step: 0.05, min: 0, max: 1 },
    },
    _run: {
      label: "Run",
      type: "button",
    },
    _section_layers: {
      label: "Layers",
      title: "Overlay display",
      type: "section",
      collapsed: false,
    },
    _show_qc: {
      label: "QC score",
      type: "checkbox",
      default: true,
    },
    _show_stripe: {
      label: "Stripe",
      type: "checkbox",
      default: false,
    },
    _show_bubble: {
      label: "Bubble",
      type: "checkbox",
      default: false,
    },
    _show_fold: {
      label: "Fold",
      type: "checkbox",
      default: false,
    },
  },
};

WSI_QC.init = function (container) {
  interfaceUtils.alert("WSI QC plugin loaded");
};

WSI_QC.inputTrigger = function (input) {
  if (input === "_run") {
    let payload = {
      wsi: WSI_QC.get("_wsi"),
      out: WSI_QC.get("_out"),
      ref: WSI_QC.get("_ref"),
      tile: WSI_QC.get("_tile"),
      qc_thresh: WSI_QC.get("_qc_thresh"),
      show: {
        qc: WSI_QC.get("_show_qc"),
        stripe: WSI_QC.get("_show_stripe"),
        bubble: WSI_QC.get("_show_bubble"),
        fold: WSI_QC.get("_show_fold"),
      },
    };

    let success = function (resp) {
      interfaceUtils.alert(resp);
      if (payload.show.qc) {
        interfaceUtils.addLayerFromUrl({
          url: payload.out + "/overlay_qc.csv",
          name: "QC score",
          type: "csv",
        });
      }
      if (payload.show.stripe) {
        interfaceUtils.addLayerFromUrl({
          url: payload.out + "/overlay_stripe.csv",
          name: "Stripe",
          type: "csv",
        });
      }
      if (payload.show.bubble) {
        interfaceUtils.addLayerFromUrl({
          url: payload.out + "/overlay_bubble.csv",
          name: "Bubble",
          type: "csv",
        });
      }
      if (payload.show.fold) {
        interfaceUtils.addLayerFromUrl({
          url: payload.out + "/overlay_fold.csv",
          name: "Fold",
          type: "csv",
        });
      }
    };

    let error = function (err) {
      console.log("WSI_QC error", err);
      interfaceUtils.alert("WSI QC failed");
    };

    WSI_QC.api("run_pipeline", payload, success, error);
  }
};
