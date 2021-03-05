import geopandas
import matplotlib.pyplot as plt
from dms2dec.dms_convert import dms2dec
# Need to install: dms2dec==0.1, geopandas==0.8.2, matplotlib==3.2.2

world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

def dms2dec_latlon(txt):
    # sample text input: '''1°34'60'N 104°28'00'E'''
    long_lat = txt.split(' ')
    return dms2dec(long_lat[0]), dms2dec(long_lat[1])

def get_geodataframe(df, geovalue):
    dfc = df.copy()
    dfc['latitude'], dfc['longitude'] = zip(*df[geovalue].map(dms2dec_latlon))
    gdf = geopandas.GeoDataFrame(
        dfc, geometry=geopandas.points_from_xy(dfc.longitude, dfc.latitude))
    return gdf

def plot_map(df, geocolumn, targetcolumn, title=None):
    color = 'blue'
    if 'No' in df[targetcolumn].unique():
        color = 'red'
    gdf = get_geodataframe(df, geocolumn)
    with plt.style.context(('seaborn', 'ggplot')): # add gray background
        # Restrict to South America.
        ax = world[(world.name!='Antarctica')].plot(figsize=(15, 8), color='white', edgecolor='gray')
        # Plot our ``GeoDataFrame``.
        gdf.plot(ax=ax, color=color, markersize=20, alpha=0.4)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(title)
        plt.show()
