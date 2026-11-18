# optimization
steps:
download file from https://apps.nationalmap.gov/downloader/#/elevation
    curl the file directly to download once you get download link (like curl -L -o data/boston_dem.tif "LINK")

python scripts/build_boston_graph.py --dem PATH_TO_ELEVATION.tif
python scripts/solve_boston_loop.py --km 3