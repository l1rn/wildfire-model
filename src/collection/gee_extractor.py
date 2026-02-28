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
            
        self.bbox = ee.Geometry.BBox(59.0, 58.0, 86.0, 65.8)
            
    def run_gee_pipeline(self):
        lc = ee.Image("ESA/WorldCover/v100/2020").clip(self.bbox).uint8()

        dem_col = ee.ImageCollection("COPERNICUS/DEM/GLO30") \
            .filterBounds(self.bbox) \
           
        native_proj = dem_col \
            .first() \
            .select('DEM') \
            .projection() 
            
        dem = dem_col \
            .mosaic() \
            .setDefaultProjection(native_proj) \
            .clip(self.bbox)
            
        ds = dem.select('DEM')
            
        terrain = ee.Terrain.products(ds) \
            .select(['DEM', 'slope']) \
            .rename(['elevation', 'slope']) \
            .float()
        
        ghm = ee.Image("CSP/HM/GlobalHumanModification/2016") \
            .select('gHM').clip(self.bbox)
            
        export_params = {
            'region': self.bbox.getInfo()['coordinates'],
            'scale': 1000,
            'crs': 'EPSG:4326',
            'fileFormat': 'GeoTIFF',
            'maxPixels': 1e9,
            'folder': 'GEE_KHMAO_RAW'
        }
        
        layers = {
            'Terrain': terrain,
        }
        
        print("Submitting tasks to GEE")
        for name, image in layers.items():
            task = ee.batch.Export.image.toDrive(
                image=image,
                description=f'KHMAO_{name}_1km',
                fileNamePrefix=f'khmao_{name}_1km',
                **export_params
            )
            task.start()
            print(f" - Started {name}")
    
    def monthly_image(self):
        start_year = 2016
        end_year = 2026
        
        years = ee.List.sequence(start_year, end_year)
        months = ee.List.sequence(1, 12)
        
        def make_monthly(y):
            y = ee.Number(y)
            
            def make_image(m):
                m = ee.Number(m)
                start = ee.Date.fromYMD(y, m, 1)
                end = start.advance(1, 'month')
                
                collection = (
                    ee.ImageCollection("MODIS/061/MOD13A1")
                    .filterDate(start, end)
                    .filterBounds(self.bbox)
                    .select('NDVI')
                )
                
                count = collection.size()
                image = ee.Image(
                    ee.Algorithms.If(
                        count.gt(0),
                        collection.mean().multiply(0.0001).toFloat(),
                        ee.Image(0).constant(-9999).toFloat()
                    )
                ).clip(self.bbox)
                
                band_name = ee.String('NDVI_') \
                    .cat(y.int().format()) \
                    .cat('_') \
                    .cat(m.format("%02d"))
            
                return image.rename([band_name])
            return months.map(make_image)
        monthly_images = years.map(make_monthly).flatten()
        monthly_collection = ee.ImageCollection(monthly_images)
        
        stacked_image = monthly_collection.toBands()
        print("Submitting single multiband export...")  
        task = ee.batch.Export.image.toDrive(
            image=stacked_image,
            description='KHMAO_NDVI_monthly_2016_2026',
            fileNamePrefix='khmao_ndvi_monthly_2016_2026',
            region=self.bbox,
            scale=500,
            crs='EPSG:4326',
            maxPixels=1e13,
            folder='GEE_KHMAO_RAW'
        )
        task.start()
        print("Monthly multiband export started")
        
    def run(self):
        self.initialize()    
        ans = int(input("choose (1 / 0): "))
        if ans == 0:
            self.run_gee_pipeline()
        elif ans == 1:
            self.monthly_image()