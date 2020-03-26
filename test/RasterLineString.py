from osgeo import gdal, ogr
import os, sys
from PIL import Image
import numpy as np

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(root_dir, "src")
sys.path.insert(0, src_dir)
# change cwd to root dir
os.chdir(root_dir)

if __name__ == '__main__':
    pixel_size = 1.0
    # Filename of input OGR file
    vector_fn = 'data/FYPData/spacenet/AOI_2_Vegas_Train/roads/SN3_roads_train_AOI_2_Vegas_geojson_roads_speed_img1.geojson'

    # Open the data source and read in the extent
    source_ds = ogr.Open(vector_fn)
    source_layer = source_ds.GetLayer()
    x_min, x_max, y_min, y_max = source_layer.GetExtent()

    # Create the destination data source
    x_res = int((x_max - x_min) / pixel_size)
    y_res = int((y_max - y_min) / pixel_size)
    target_ds = gdal.GetDriverByName('MEM').Create("", 650, 650, 3, gdal.GDT_Byte)
    target_ds.SetGeoTransform((x_min, (x_max - x_min) / 650, 0, y_max, 0, -(y_max - y_min) / 650))
    band = target_ds.GetRasterBand(1)

    # Rasterize
    gdal.RasterizeLayer(target_ds, [1,2,3], source_layer, burn_values=[0,0,255])
    img = target_ds.ReadAsArray()
    img = np.moveaxis(img, 0, -1)
    Image.fromarray(img).show()
