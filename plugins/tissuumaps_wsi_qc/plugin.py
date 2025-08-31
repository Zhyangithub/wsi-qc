# Placeholder TissUUmaps plugin (to be wired to TissUUmaps plugin API)
# For v0 we will load CSV overlays directly in TissUUmaps without this plugin.
# In v1, expose controls to run QC on the fly and render heatmaps.
class Plugin:
    NAME = "WSI QC"
    DESCRIPTION = "Stain normalization + artifact QC heatmaps for WSI"
    VERSION = "0.0.1"

    def __init__(self):
        pass

    def run(self):
        # TODO: hook into TissUUmaps' UI callbacks
        print("WSI-QC plugin placeholder")
