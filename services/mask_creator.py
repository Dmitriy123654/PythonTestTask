import numpy as np
from PIL import Image
from geojson import MultiPolygon
from rasterio.features import shapes
from shapely.geometry import Polygon
from shapely.ops import unary_union
from typing import Dict
import geojson

def create_mask(image_array: np.ndarray) -> np.ndarray:
    """
    Создание маски полезной области.
    Пиксели с яркостью 0 или 255 считаются неинформативными (фон).
    Полезные области имеют значение 1, фон — 0.
    """
    if image_array.ndim == 3:
        image = Image.fromarray(image_array)
        image_array = np.array(image.convert("L"))

    mask = np.where((image_array > 0) & (image_array < 255), 1, 0).astype(np.uint8)
    return mask


def create_polygon_from_mask(mask: np.ndarray) -> Dict:
    """
    Создает векторный полигон (GeoJSON) из маски.
    """
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    shapes_generator = shapes(mask, mask=mask > 0)
    polygons = []

    for shape_data, value in shapes_generator:
        if value == 1:
            polygons.append(Polygon(shape_data['coordinates'][0]))

    if polygons:
        merged_polygon = unary_union(polygons)
        if isinstance(merged_polygon, Polygon):
            merged_polygon = [merged_polygon]
        elif isinstance(merged_polygon, MultiPolygon):
            merged_polygon = list(merged_polygon.geoms)

        return geojson.FeatureCollection(
            [geojson.Feature(geometry=poly.__geo_interface__) for poly in merged_polygon]
        )
    else:
        print("No polygons created.")
        return geojson.FeatureCollection([])