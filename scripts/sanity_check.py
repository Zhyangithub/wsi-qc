import sys, importlib, json

mods = [
    "numpy", "pandas", "cv2", "skimage", "openslide", "tqdm",
    "anndata", "scanpy", "squidpy", "matplotlib", "sklearn"
]

report = {}
for m in mods:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "unknown")
        report[m] = {"ok": True, "version": ver}
    except Exception as e:
        report[m] = {"ok": False, "error": str(e)}

print(json.dumps(report, indent=2))

# Optional: quick openslide capability check
try:
    import openslide
    print("OpenSlide vendor:", openslide.__library_name__)
    print("OpenSlide version:", openslide.__library_version__)
except Exception as e:
    print("OpenSlide check failed:", e, file=sys.stderr)
