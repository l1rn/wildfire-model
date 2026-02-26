import ee

class GeeExtractor:
    def __init__(self):
        self.bbox = None
        self.project_name = "siberian-487118"
        
    def initialize(self):
        try:
            ee.Initialize(project=self.project_name)
        except Exception:
            ee.Authenticate()
            ee.Initialize(project=self.project_name)
            
        self.bbox = ee.Geometry.BBox(59.0, 59.0, 78.0, 64.0)
            
    def run_gee_pipeline(self):
        lc = ee.Image("ESA/WorldCover/v100/2020").clip(self.bbox).uint8()

        dem = ee.Image("USGS/SRTMGL1_003").clip(self.bbox)
        terrain = ee.Terrain.products(dem) \
            .select(['elevation', 'slope']).clip(self.bbox).float()
        
        ghm = ee.Image("CSP/HM/GlobalHumanModification/2016") \
            .select('gHM').clip(self.bbox)
            
        export_params = {
            'region': self.bbox.getInfo()['coordinates'],
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
    
    def monthly_image(self):
        start_year = 2016
        end_year = 2026
        
        years = ee.List.sequence(start_year, end_year)
        months = ee.List.sequence(1, 12)
        modis = ee.ImageCollection("MODIS/061/MOD13A1") \
            .select(['NDVI', 'EVI']) \
            .filterDate('2016-01-01', '2026-01-01') \
            .filterBounds(self.bbox)
            
        modis_image = modis.mean().multiply(0.0001)
        
        export_params = {
            'region': self.bbox.getInfo()['coordinates'],
            'crs': 'EPSG:4326',
            'maxPixels': 1e9,
            'folder': 'GEE_KHMAO_RAW'
        }
        print("Submitting the task to GEE")
        task = ee.batch.Export.image.toDrive(
            image=modis_image,
            description=f'KHMAO_MOD13A1_90m',
            fileNamePrefix=f'khmao_mod13a1_90m',
            **export_params
        )
        task.start()
        print("- Modis Task Started")
        
    def run(self):
        self.initialize()    
        ans = int(input("choose (1 / 0): "))
        if ans == 0:
            self.run_gee_pipeline()
        elif ans == 1:
            self.run_modis_pipeline()