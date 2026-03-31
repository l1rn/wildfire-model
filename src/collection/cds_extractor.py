import cdsapi
import os

def extract_era5():
    c = cdsapi.Client(
        url="https://cds.climate.copernicus.eu/api",
        key="7df8a456-50e8-472c-8990-d312fc2dde02" 
    )
    
    target_years = [str(y) for y in range(2010, 2025)]
    wildfire_months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'] 
    khmao_area = [65.8, 58.0, 57.0, 86.0]

    print("Requesting ERA5 Hourly Data from Copernicus...")

    for year in target_years:
        filename = f'data/raw/era5_hourly_khmao_{year}.nc'
        
        if os.path.exists(filename):
            print(f"File {filename} already exists, skipping...")
            continue
            
        print(f"Downloading data for {year}...")
        try:
            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'variable': [
                        '2m_temperature', 
                        '2m_dewpoint_temperature',
                        '10m_u_component_of_wind', 
                        '10m_v_component_of_wind',
                        'total_precipitation', 
                        'volumetric_soil_water_layer_1'
                    ],
                    'year': [year],
                    'month': wildfire_months,
                    'day': [
                        '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                        '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                        '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31'
                    ],
                    'time': [
                        '00:00', '03:00', '06:00', '09:00', 
                        '12:00', '15:00', '18:00', '21:00'
                    ], 
                    'area': khmao_area,
                },
                filename
            )
        except Exception as e:
            print(f"Failed on year {year}: {e}")

    print("All downloads complete!")