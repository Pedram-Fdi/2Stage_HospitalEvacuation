import geopandas as gpd
import contextily as cx
import matplotlib.pyplot as plt
import os
from shapely.geometry import Point

# -------------------------------
# Coordinate Conversion Function
# -------------------------------
def convert_to_latlon(points, x_range=(0, 50), y_range=(0, 50),
                      lat_range=(45.45, 45.65), lon_range=(-73.90, -73.45)):
    x_min, x_max = x_range
    y_min, y_max = y_range
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range

    converted = []
    for x, y in points:
        lat = lat_min + (y - y_min) / (y_max - y_min) * (lat_max - lat_min)
        lon = lon_min + (x - x_min) / (x_max - x_min) * (lon_max - lon_min)
        converted.append((lat, lon))
    return converted

# -------------------------------
# Plotting Function on Real Map
# -------------------------------
def plot_facilities_on_map(instance, solution=None, only_open_acfs=False, filename="output_map.png", fixed_bounds=None):
    # Convert coordinates to lat/lon
    instance.DisasterArea_Position = convert_to_latlon(instance.DisasterArea_Position)
    instance.Hospital_Position = convert_to_latlon(instance.Hospital_Position)
    instance.ACF_Position = {
        i: latlon for i, latlon in zip(instance.ACFSet, convert_to_latlon([instance.ACF_Position[i] for i in instance.ACFSet]))
    }

    points = []

    # Always add Disaster Areas and Hospitals (they should appear in both maps)
    if not only_open_acfs:
        # Disaster Areas
        for (lat, lon) in instance.DisasterArea_Position:
            points.append({'geometry': Point(lon, lat), 'type': 'Disaster Area'})

        # Hospitals
        for (lat, lon) in instance.Hospital_Position:
            points.append({'geometry': Point(lon, lat), 'type': 'Hospital'})

        # All ACFs
        for i in instance.ACFSet:
            lat, lon = instance.ACF_Position[i]
            points.append({'geometry': Point(lon, lat), 'type': 'ACF'})
    else:
        # For open ACFs only map, still show disaster areas and hospitals
        for (lat, lon) in instance.DisasterArea_Position:
            points.append({'geometry': Point(lon, lat), 'type': 'Disaster Area'})

        # Hospitals
        for (lat, lon) in instance.Hospital_Position:
            points.append({'geometry': Point(lon, lat), 'type': 'Hospital'})
        
        # Only open ACFs
        open_acf_count = 0
        total_acf_count = len(instance.ACFSet)
        
        print(f"Debug: Checking {total_acf_count} ACFs for open status...")
        
        for i in instance.ACFSet:
            acf_status = solution.ACFEstablishment_x_wi[0][i] if solution else 0
            print(f"Debug: ACF {i} status: {acf_status}")
            
            if solution and solution.ACFEstablishment_x_wi[0][i] == 1:
                lat, lon = instance.ACF_Position[i]
                points.append({'geometry': Point(lon, lat), 'type': 'ACF'})
                open_acf_count += 1
                print(f"Debug: Added open ACF {i} at ({lat}, {lon})")
        
        print(f"Debug: Found {open_acf_count} open ACFs out of {total_acf_count} total ACFs")
        
        # If no ACFs are open, let's add a test point to see if the issue is visibility
        if open_acf_count == 0:
            print("Debug: No open ACFs found! Adding test ACF point...")
            # Add the first ACF as a test
            first_acf = list(instance.ACFSet)[0]
            lat, lon = instance.ACF_Position[first_acf]
            points.append({'geometry': Point(lon, lat), 'type': 'ACF'})
            print(f"Debug: Added test ACF at ({lat}, {lon})")

    print(f"Debug: Total points to plot: {len(points)}")
    
    # Create GeoDataFrame and convert to Web Mercator
    gdf = gpd.GeoDataFrame(points, crs='EPSG:4326').to_crs(epsg=3857)
    
    print(f"Debug: GeoDataFrame bounds: {gdf.total_bounds}")
    print(f"Debug: Sample coordinates after projection:")
    for i, row in gdf.head(3).iterrows():
        print(f"  {row['type']}: ({row.geometry.x:.2f}, {row.geometry.y:.2f})")

    # Plot with basemap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Add basemap FIRST (so points appear on top)
    if fixed_bounds:
        print(f"Debug: Using fixed bounds: {fixed_bounds}")
        ax.set_xlim(fixed_bounds['x_min'], fixed_bounds['x_max'])
        ax.set_ylim(fixed_bounds['y_min'], fixed_bounds['y_max'])
    else:
        # Let geopandas set the bounds naturally first
        bounds = gdf.total_bounds
        padding = 1000
        ax.set_xlim(bounds[0] - padding, bounds[2] + padding)
        ax.set_ylim(bounds[1] - padding, bounds[3] + padding)
        print(f"Debug: Using natural bounds with padding: xlim=({bounds[0] - padding:.2f}, {bounds[2] + padding:.2f}), ylim=({bounds[1] - padding:.2f}, {bounds[3] + padding:.2f})")
    
    cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik, alpha=0.5)
    
    # Plot points ON TOP of basemap with higher zorder
    points_plotted = {}
    for marker_type, marker_style in {
        'Disaster Area': {'color': 'red'},
        'Hospital': {'color': 'blue'},
        'ACF': {'color': 'green'},
    }.items():
        subset = gdf[gdf['type'] == marker_type]
        if not subset.empty:
            print(f"Debug: Plotting {len(subset)} {marker_type} points")
            
            # Print actual coordinates being plotted
            print(f"Debug: {marker_type} coordinates (first 3):")
            for i, row in subset.head(3).iterrows():
                print(f"  ({row.geometry.x:.2f}, {row.geometry.y:.2f})")
            
            subset.plot(ax=ax, color=marker_style['color'], markersize=100, 
                       label=marker_type, zorder=10, edgecolor='white', linewidth=2,
                       alpha=0.8)
            points_plotted[marker_type] = len(subset)
        else:
            print(f"Debug: No {marker_type} points to plot")
            points_plotted[marker_type] = 0
    
    print(f"Debug: Points plotted summary: {points_plotted}")
    print(f"Debug: Final axis limits: xlim={ax.get_xlim()}, ylim={ax.get_ylim()}")
    
    # Add title to distinguish the maps
    map_title = "All Facilities" if not only_open_acfs else "Open ACFs Only"
    ax.set_title(map_title, fontsize=14, fontweight='bold')

    ax.set_axis_off()
    plt.legend()
    plt.tight_layout()

    output_dir = "UI/Solution_UI"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Return the bounds for consistency across calls
    if not fixed_bounds:
        bounds = ax.get_xlim() + ax.get_ylim()
        return {
            'x_min': bounds[0],
            'x_max': bounds[1], 
            'y_min': bounds[2],
            'y_max': bounds[3]
        }
    return fixed_bounds

def get_all_points_bounds(instance):
    """Get bounds that encompass all points (disaster areas, hospitals, and ACFs)"""
    # Convert coordinates to lat/lon
    disaster_coords = convert_to_latlon(instance.DisasterArea_Position)
    hospital_coords = convert_to_latlon(instance.Hospital_Position)
    acf_coords = convert_to_latlon([instance.ACF_Position[i] for i in instance.ACFSet])
    
    # Combine all coordinates
    all_coords = disaster_coords + hospital_coords + acf_coords
    
    print(f"Debug: All coordinates range:")
    lats = [coord[0] for coord in all_coords]
    lons = [coord[1] for coord in all_coords]
    print(f"  Latitude: {min(lats):.6f} to {max(lats):.6f}")
    print(f"  Longitude: {min(lons):.6f} to {max(lons):.6f}")
    
    # Create points and convert to Web Mercator
    points = [Point(lon, lat) for lat, lon in all_coords]
    gdf = gpd.GeoDataFrame({'geometry': points}, crs='EPSG:4326').to_crs(epsg=3857)
    
    # Get bounds with some padding (use smaller padding for small areas)
    bounds = gdf.total_bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    padding = max(width, height) * 0.1  # 10% padding
    
    print(f"Debug: Bounds in Web Mercator: {bounds}")
    print(f"Debug: Using padding: {padding:.2f} meters")
    
    return {
        'x_min': bounds[0] - padding,
        'x_max': bounds[2] + padding,
        'y_min': bounds[1] - padding,
        'y_max': bounds[3] + padding
    }