import ee

def run_gee_pipeline():
    
    try:
        ee.Initialize(project="siberian-487118")
    except Exception:
        ee.Authenticate()
        ee.Initialize(project="siberian-487118")
        
    bbox = ee.Geometry.BBox(59.0, 59.0, 78.0, 64.0)
    
    lc = ee.Image("ESA/WorldCover/v100/2020").clip(bbox).uint8()

    dem = ee.Image("USGS/SRTMGL1_003").clip(bbox)
    terrain = ee.Terrain.products(dem).select(['elevation', 'slope']).clip(bbox).float()
    
    ghm = ee.Image("CSP/HM/GlobalHumanModification/2016").select('gHM').clip(bbox)
        
    export_params = {
        'region': bbox.getInfo()['coordinates'],
        'scale': 90,
        'crs': 'EPSG:4326',
        'fileFormat': 'GeoTIFF',
        'maxPixels': 1e9,
        'folder': 'GEE_KHMAO_RAW'
    }
    
    layers = {
        'Landcover': lc,
        'Terrain': terrain,
        'Human_Mod': ghm,
    }
    
    print("Submitting tasks to GEE")
    for name, image in layers.items():
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=f'KHMAO_{name}_90m',
            fileNamePrefix=f'khmao_{name}_90m',
            **export_params
        )
        task.start()
        print(f" - Started {name}")
        