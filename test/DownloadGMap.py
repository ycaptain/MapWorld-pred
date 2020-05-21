import os, sys, re
from pathlib import Path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(root_dir, "src")
sys.path.insert(0, src_dir)
# change cwd to root dir
os.chdir(root_dir)

from utils import util_geo


def main():
    spacenet_path = Path("data/FYPData/spacenet/buildings/AOI_2_Vegas_Train/RGB-PanSharpen")
    sat2map_out_path = Path("data/FYPData/sat2map_spacenet")
    s2m_A = sat2map_out_path / "trainA"
    s2m_B = sat2map_out_path / "trainB"
    s2m_A.mkdir(parents=True, exist_ok=True)
    s2m_B.mkdir(parents=True, exist_ok=True)

    util = util_geo.GeoLabelUtil.GeoImgUtil()
    for img_name in os.listdir(spacenet_path):
        img_src = spacenet_path / img_name
        img = util.load_geotiff(img_src)
        rimg = util.normalize_img(img.ReadAsArray())

        # change to .png
        re_img_index = re.compile("img(\d+)")
        idx = re_img_index.search(img_name).group(1)
        # filename, file_extension = os.path.splitext(img_name)

        tg_Aimg = s2m_A / ("%d_A.jpg" % int(idx))
        tg_Bimg = s2m_B / ("%d_B.jpg" % int(idx))

        util.save_rgb(rimg, tg_Aimg)
        print("Satellite saved to", tg_Aimg)

        ulx, xres, xskew, uly, yskew, yres = img.GetGeoTransform()
        # (-115.1706276, 2.7000000000043656e-06, 0.0, 36.2406177, 0.0, -2.7000000000043656e-06)
        cx = ulx + (img.RasterXSize * xres / 2)
        cy = uly + (img.RasterYSize * yres / 2)

        util.download_from_gmap(tg_Bimg, cy, cx)
        print("Map saved to", tg_Bimg)


if __name__ == '__main__':
    main()
