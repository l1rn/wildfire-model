import ee
import pandas as pd

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
            
        ogim = ee.FeatureCollection("EDF/OGIM/current") \
            .filterBounds(self.bbox)
            
        dist_oil_gas = ogim.distance(100000) \
            .clip(self.bbox) \
            .divide(1000).rename('dist_oil_gas_km').float()
            
        grip_asia = ee.FeatureCollection("projects/sat-io/open-datasets/GRIP4/Middle-East-Central-Asia")
        roads = grip_asia.filterBounds(self.bbox)
        roads_count = roads.reduceToImage([], ee.Reducer.countEvery()).unmask(0)
        road_density = roads_count.focalMean(radius=5000, units='meters') \
            .rename("road_density_5km").float()
        
        peat = ee.Image("projects/sat-io/open-datasets/ML-GLOBAL-PEATLAND-EXTENT") \
            .clip(self.bbox).rename("peatland_flag").unmask()
            
        pop_density = ee.ImageCollection("JRC/GHSL/P2023A/GHS_POP") \
            .sort('system:time_start', False) \
            .first().rename("pop_density").clip(self.bbox)
            
        export_params = {
            'region': self.bbox.getInfo()['coordinates'],
            'scale': 1000,
            'crs': 'EPSG:4326',
            'fileFormat': 'GeoTIFF',
            'maxPixels': 1e9,
            'folder': 'GEE_KHMAO_RAW'
        }
        
        layers = {
            'PopDensity': pop_density
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
        
    def validate_with_sentinel2(self, csv_path):
        self.initialize()    
        
        df = pd.read_csv(csv_path)
        results = []
        print(f"Starting validatin for {len(df)} points...")

        for index, row in df.iterrows():
            lon, lat = row['longitude'], row['latitude']
            date_str = row['acq_date']
            fire_type = row['type']
            
            point = ee.Geometry.Point([lon, lat])
            
            roi = point.buffer(2000).bounds()
            fire_date = ee.Date(date_str)
            try:
                s2_col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
                    .filterBounds(roi) \
                    .filterDate(fire_date.advance(-60, 'day'), fire_date.advance(60, 'day')) \
                    .sort('CLOUDY_PIXEL_PERCENTAGE')
                    
                count = s2_col.size().getInfo()
                if count == 0:
                    print(f"No imagery found for date {date_str}")
                    continue
                    
                pre = s2_col.filterDate(fire_date.advance(-60, 'day'), fire_date).median()
                post = s2_col.filterDate(fire_date, fire_date.advance(460, 'day')).median()
                
                def get_nbr(img):
                    return img.normalizedDifference(['B8', 'B12'])
                
                nbr_pre = get_nbr(pre)
                nbr_post = get_nbr(post)
                dnbr = nbr_pre.subtract(nbr_post)

                rbr = dnbr.divide(nbr_pre.add(1.001).abs().sqrt())
                burned_mask = rbr.gt(0.1)
                stats = burned_mask.multiply(ee.Image.pixelArea()).reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=roi,
                    scale=10,
                    maxPixels=1e9
                )
                
                s2_area_ha = ee.Number(stats.get('nd')).divide(10000).getInfo()

                results.append({
                    'lat': lat, 'lon': lon, 'date': date_str, 
                    'viirs_type': fire_type, 's2_burned_ha': s2_area_ha
                })
                print(f"[{index}] Type {fire_type} at {date_str}: S2 Area = {s2_area_ha:.2f} ha")

            except Exception as e:
                print(f"Error at index {index}: {e}")
                continue
            
        res_df = pd.DataFrame(results)
        res_df.to_csv('validation_results.csv', index=False)
        return res_df
    def run(self):
        self.initialize()    
        ans = int(input("choose (1 / 0): "))
        if ans == 0:
            self.run_gee_pipeline()
        elif ans == 1:
            self.monthly_image()