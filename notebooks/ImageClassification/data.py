import os
import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.windows import Window
from PIL import Image
import pystac_client
import planetary_computer
from datetime import datetime, timedelta
import calendar
import warnings
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings('ignore')

IMAGE_SIZE = 224
BANDS = ['B04', 'B03', 'B02']  

class SentinelImageProcessor:
    def __init__(self):
        self.catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    
    def get_sentinel_imagery(self, bbox, date_range, cloud_cover_max=20):
        try:
            
            search = self.catalog.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox,
                datetime=f"{date_range[0]}/{date_range[1]}",
                query={"eo:cloud_cover": {"lt": cloud_cover_max}}
            )
            
            items = search.get_all_items()
            
            if len(items) == 0:
                print(f"No Sentinel-2 data found for bbox {bbox} during {date_range}")
                return None, None
            
            
            items = sorted(items, key=lambda x: (x.properties.get("eo:cloud_cover", 100), 
                                               x.properties.get("datetime")))
            
            item = items[0]
            item_datetime = item.properties.get("datetime").split("T")[0]
            cloud_cover = item.properties.get("eo:cloud_cover")
            
            print(f"✓ Found Sentinel-2 data for {date_range} (date: {item_datetime}, cloud cover: {cloud_cover}%)")
            
            
            metadata = {
                "datetime": item.properties.get("datetime"),
                "cloud_cover": cloud_cover,
                "platform": item.properties.get("platform"),
                "item_id": item.id
            }
            
            
            signed_item = planetary_computer.sign(item)
            
            
            rgb_bands = []
            for band in BANDS:
                asset = signed_item.assets.get(band)
                if asset:
                    url = asset.href
                    with rasterio.open(url) as src:
                        
                        height, width = src.shape
                        center_x, center_y = width // 2, height // 2
                        half_size = min(width, height, 2 * IMAGE_SIZE) // 2
                        window = Window(
                            center_x - half_size, 
                            center_y - half_size,
                            2 * half_size,
                            2 * half_size
                        )
                        band_data = src.read(1, window=window)
                        
                        
                        if band_data.shape[0] != IMAGE_SIZE or band_data.shape[1] != IMAGE_SIZE:
                            band_data = np.array(Image.fromarray(band_data).resize((IMAGE_SIZE, IMAGE_SIZE)))
                        
                        rgb_bands.append(band_data)
                else:
                    print(f"Band {band} not found in assets")
                    return None, None
            
            
            rgb_image = np.stack(rgb_bands, axis=0)
            
            
            rgb_image = rgb_image.astype(np.float32) / 10000.0
            
            return rgb_image, metadata
            
        except Exception as e:
            print(f"Error downloading Sentinel-2 imagery: {e}")
            return None, None

class EBirdDataProcessor:
    def __init__(self, api_key=None):
        
        self.api_key = api_key or os.getenv("EBIRD_API_KEY")
        if not self.api_key:
            raise ValueError("eBird API key is required. Set it as EBIRD_API_KEY in your .env file or pass it to the constructor.")
    
    def get_ebird_data(self, bbox, date_range, species_list):
        try:
            
            start_date = datetime.strptime(date_range[0], "%Y-%m-%d")
            end_date = datetime.strptime(date_range[1], "%Y-%m-%d")
            
            
            mid_date = start_date + (end_date - start_date) / 2
            month = mid_date.month
            year = mid_date.year
            month_name = calendar.month_name[month]
            
            
            center_lat = (bbox[1] + bbox[3]) / 2
            center_lon = (bbox[0] + bbox[2]) / 2
            
            
            url = "https://api.ebird.org/v2/data/obs/geo/recent"
            
            params = {
                "lat": center_lat,
                "lng": center_lon,
                "dist": 10,  
                "back": 30,  
                "includeProvisional": "true"
            }
            
            headers = {
                "X-eBirdApiToken": self.api_key
            }
            
            
            response = requests.get(url, params=params, headers=headers)
            
            if response.status_code != 200:
                print(f"Error fetching eBird data: {response.status_code}")
                return [0] * len(species_list), None
            
            
            observations = response.json()
            
            
            species_presence = [0] * len(species_list)
            species_counts = [0] * len(species_list)
            first_observed = [None] * len(species_list)
            
            
            for obs in observations:
                species_code = obs.get("speciesCode")
                obs_date = obs.get("obsDt", "").split("T")[0]  
                
                if species_code in species_list:
                    index = species_list.index(species_code)
                    species_presence[index] = 1
                    species_counts[index] += 1
                    
                    
                    if first_observed[index] is None or obs_date < first_observed[index]:
                        first_observed[index] = obs_date
            
            
            species_found = sum(species_presence)
            if species_found > 0:
                found_species = [species_list[i] for i in range(len(species_list)) if species_presence[i] == 1]
                print(f"✓ Found {species_found} bird species in area {bbox}: {', '.join(found_species)}")
            else:
                print(f"✓ Processed eBird data for area {bbox}, but no target species found")
            
            
            metadata = {
                "total_observations": len(observations),
                "species_found": species_found,
                "observation_period": f"{date_range[0]} to {date_range[1]}",
                "month": month_name,
                "year": year,
                "species_counts": {species_list[i]: species_counts[i] for i in range(len(species_list))},
                "first_observed": {species_list[i]: first_observed[i] for i in range(len(species_list)) if first_observed[i] is not None}
            }
            
            return species_presence, metadata
            
        except Exception as e:
            print(f"Error processing eBird data: {e}")
            return [0] * len(species_list), None

def generate_dataset(locations, date_ranges, species_list, ebird_api_key=None):
    image_processor = SentinelImageProcessor()
    
    ebird_api_key = ebird_api_key or os.getenv("EBIRD_API_KEY")
    if not ebird_api_key:
        raise ValueError("eBird API key is required. Set it as EBIRD_API_KEY in your .env file or pass it to the function.")
    
    ebird_processor = EBirdDataProcessor(ebird_api_key)
    
    images = []
    labels = []
    metadata = []
    
    print("Generating dataset...")
    
    
    total_combinations = len(locations) * len(date_ranges)
    progress_count = 0
    
    for bbox in locations:
        for date_range in date_ranges:
            progress_count += 1
            print(f"\nProcessing location {bbox} for period {date_range} ({progress_count}/{total_combinations})")
            
            
            image, image_meta = image_processor.get_sentinel_imagery(bbox, date_range)
            
            if image is not None:
                
                bird_presence, bird_meta = ebird_processor.get_ebird_data(bbox, date_range, species_list)
                
                if bird_meta is not None:
                    
                    images.append(image)
                    labels.append(bird_presence)
                    
                    
                    combined_meta = {
                        "location": bbox,
                        "date_range": date_range,
                        "image_metadata": image_meta,
                        "bird_metadata": bird_meta
                    }
                    
                    metadata.append(combined_meta)
                    
                    
                    present_species = sum(bird_presence)
                    print(f"+ Successfully added sample with {present_species} species present")
    
    print(f"\nDataset generation complete: {len(images)} samples collected")
    
    if len(images) == 0:
        print("Warning: No samples were collected. Check API keys and data availability.")
        return np.array([]), np.array([]), []
        
    return np.array(images), np.array(labels), metadata