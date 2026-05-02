import ee
import pandas as pd

import questionary

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
        lc = ee.Image("ESA/WorldCover/v200/2021").clip(self.bbox).uint8()

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
                
        ghm = ee.ImageCollection("CSP/HM/GlobalHumanModification") \
            .first().clip(self.bbox)
            
        ogim = ee.FeatureCollection("EDF/OGIM/current") \
            .filterBounds(self.bbox)
            
        dist_oil_gas = ogim.distance(10000) \
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
        
        cisi = ee.Image("projects/sat-io/open-datasets/CISI/global_CISI") \
            .clip(self.bbox).rename("cisi")
            
        years = range(2010, 2024) 
        
        yearly_bands = []
        
        for year in years:
            base_empty = ee.Image.constant(0).byte().rename(f'burned_{year}')
            fires = ee.ImageCollection("MODIS/061/MCD64A1") \
                .filterBounds(self.bbox) \
                .filterDate(f'{year}-01-01', f'{year}-12-31') \
                .select('BurnDate')
            
            fire_max = fires.max().gt(0).rename(f'burned_{year}').byte()
            
            yearly_burned = ee.ImageCollection([base_empty, fire_max]).mosaic()
            
            yearly_bands.append(yearly_burned)
            
        burned_area_multiband = ee.Image.cat(yearly_bands).clip(self.bbox)
            
        export_params = {
            'region': self.bbox.getInfo()['coordinates'],
            'scale': 10000,
            'crs': 'EPSG:4326',
            'fileFormat': 'GeoTIFF',
            'maxPixels': 1e9,
            'folder': 'GEE_KHMAO_RAW'
        }
        options = questionary.checkbox(
            "Select options:",
            choices=[
                "landcover",
                "terrain",
                "ghm",
                "dist_oil_gas",
                "road_density",
                "peatland_flag",
                "pop_density",
                "burned_area_yearly",
                "cisi"
            ]
        ).ask()
        
        layer_mapping = {
            "landcover": lc,
            "terrain": terrain,
            "ghm": ghm,
            "dist_oil_gas": dist_oil_gas,
            "road_density": road_density,
            "peatland_flag": peat,
            "pop_density": pop_density,
            "burned_area_yearly": burned_area_multiband,
            "cisi": cisi
        }
        
        layers = {key: layer_mapping[key] for key in options}
        
        print("Submitting tasks to GEE")
        for name, image in layers.items():
            task = ee.batch.Export.image.toDrive(
                image=image,
                description=f'KHMAO_{name}_10km',
                fileNamePrefix=f'khmao_{name}_10km',
                **export_params
            )
            task.start()
            print(f" - Started {name}")
    
    def monthly_image(self):
        start_year = 2010
        end_year = 2024
        
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
            description='KHMAO_NDVI_monthly_2010_2024',
            fileNamePrefix='khmao_ndvi_monthly_2010_2024',
            region=self.bbox,
            crs='EPSG:4326',
            maxPixels=1e13,
            folder='GEE_KHMAO_RAW'
        )
        task.start()
        print("Monthly multiband export started")
        
    def export_lai_monthly(self):
        start_year = 2010
        end_year = 2024
        
        years = ee.List.sequence(start_year, end_year)
        months = ee.List.sequence(1, 12)
        def make_monthly(y):
            y = ee.Number(y)
            
            def make_image(m):
                m = ee.Number(m)
                start = ee.Date.fromYMD(y, m, 1)
                end = start.advance(1, 'month')
                
                collection = (
                    ee.ImageCollection("projects/sat-io/open-datasets/BU_LAI_FPAR/wgs_005degree_bimonthly")
                    .filterDate(start, end)
                    .filterBounds(self.bbox)
                    .select('LAI')
                )
                
                count = collection.size()
                
                image = ee.Image(
                    ee.Algorithms.If(
                        count.gt(0),
                        collection.mean().toFloat(),
                        ee.Image(0).constant(-9999).toFloat()
                    )
                ).clip(self.bbox)
                
                band_name = ee.String('LAI_') \
                    .cat(y.int().format()) \
                    .cat('_') \
                    .cat(m.format("%02d"))
            
                return image.rename([band_name])
            return months.map(make_image)
        
        monthly_images = years.map(make_monthly).flatten()
        monthly_collection = ee.ImageCollection(monthly_images)
        
        stacked_image = monthly_collection.toBands()
        print("Submitting LAI multiband export...")  
        task = ee.batch.Export.image.toDrive(
            image=stacked_image,
            description='KHMAO_LAI_monthly_2010_2024',
            fileNamePrefix='khmao_lai_monthly_2010_2024',
            region=self.bbox,
            crs='EPSG:4326',
            maxPixels=1e13,
            folder='GEE_KHMAO_RAW'
        )
        task.start()
        print("LAI export started! Check your Google Drive.")
    def export_fpar_monthly(self):
        start_year = 2010
        end_year = 2024
        
        years = ee.List.sequence(start_year, end_year)
        months = ee.List.sequence(1, 12)
        def make_monthly(y):
            y = ee.Number(y)
            
            def make_image(m):
                m = ee.Number(m)
                start = ee.Date.fromYMD(y, m, 1)
                end = start.advance(1, 'month')
                
                collection = (
                    ee.ImageCollection("projects/sat-io/open-datasets/BU_LAI_FPAR/wgs_005degree_bimonthly")
                    .filterDate(start, end)
                    .filterBounds(self.bbox)
                    .select('FPAR')
                )
                
                count = collection.size()
                
                image = ee.Image(
                    ee.Algorithms.If(
                        count.gt(0),
                        collection.mean().toFloat(),
                        ee.Image(0).constant(-9999).toFloat()
                    )
                ).clip(self.bbox)
                
                band_name = ee.String('FPAR_') \
                    .cat(y.int().format()) \
                    .cat('_') \
                    .cat(m.format("%02d"))
            
                return image.rename([band_name])
            return months.map(make_image)
    
        monthly_images = years.map(make_monthly).flatten()
        monthly_collection = ee.ImageCollection(monthly_images)
        
        stacked_image = monthly_collection.toBands()
        print("Submitting FPAR multiband export...")  
        task = ee.batch.Export.image.toDrive(
            image=stacked_image,
            description='KHMAO_LAI_monthly_2010_2024',
            fileNamePrefix='khmao_lai_monthly_2010_2024',
            region=self.bbox,
            crs='EPSG:4326',
            maxPixels=1e13,
            folder='GEE_KHMAO_RAW'
        )
        task.start()
        print("LAI export started! Check your Google Drive.")
      
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
        options = questionary.checkbox(
            "Select options:",
            choices=[
                "raster images (gee pipeline)",
                "ndvi monthly images",
                "lai monthly images",
                "fpar monthly images",
            ]
        ).ask()
        
        if "raster images (gee pipeline)" in options:
            self.run_gee_pipeline()
        if "ndvi monthly images" in options:
            self.monthly_image()
        if "lai monthly images" in options:
            self.export_lai_monthly()
        if "fpar monthly images" in options:
            self.export_fpar_monthly()