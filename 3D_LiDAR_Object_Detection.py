import argparse
from typing import List
import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results
import torch
from ouster.sdk import open_source
from ouster.sdk.core import ChanField, LidarScan, LidarScanSet, ScanSource, destagger, XYZLut, SensorInfo, AutoExposure
from ouster.sdk.core._utils import BeamUniformityCorrector
from ouster.sdk.viz import SimpleViz
import matplotlib as mpl
from matplotlib import cm

class SourceIterator:
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"

    def __init__(self, source: ScanSource, use_opencv=False):
        self._source: ScanSource = source
        self._use_opencv = use_opencv
        self._sensor_info: List[SensorInfo] = source.sensor_info
        self._xyzluts = [XYZLut(info) for info in self._sensor_info]
        self._generate_rgb_table()
        self.field_to_util = []
        for _ in self._sensor_info:
            self.field_to_util.append({})
            for field in [ChanField.NEAR_IR, ChanField.REFLECTIVITY]:
                self.field_to_util[-1][field] = {
                    "ae": AutoExposure(),
                    "buc": BeamUniformityCorrector(),
                    "model": YOLO("yolo26x-seg.pt").to(device=self.DEVICE)
                }
        self.name_to_class = {}
        for key, value in self.field_to_util[0][ChanField.NEAR_IR]["model"].names.items():
            self.name_to_class[value] = key
        self.classes_to_detect = [
            self.name_to_class['person'],
            self.name_to_class['car'],
            self.name_to_class['truck'],
            self.name_to_class['bus']
        ]

    @property
    def sensor_info(self) -> List[SensorInfo]:
        return self._sensor_info
    def __iter__(self):
        for scans in self._source:
            yield self.update(scans)
    def _generate_rgb_table(self):
        np.random.seed(0)
        N_COLORS = 256
        scalarMap = cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=1.0), cmap=mpl.pyplot.get_cmap('hsv'))
        self._mono_to_rgb_lut = np.clip(0.25 + 0.75 * scalarMap.to_rgba(np.random.random_sample((N_COLORS)))[:, :3], 0,1)
        self._mono_to_rgb_lut = self._mono_to_rgb_lut.astype(np.float32)

    def mono_to_rgb(self, mono_img, background_img=None):
        assert (np.issubdtype(mono_img.dtype, np.integer))
        rgb = self._mono_to_rgb_lut[mono_img % self._mono_to_rgb_lut.shape[0], :]
        if background_img is not None:
            if background_img.shape[-1] == 3:
                rgb[mono_img == 0, :] = background_img[mono_img == 0, :]
            else:
                rgb[mono_img == 0, :] = background_img[mono_img == 0, np.newaxis]
        else:
            rgb[mono_img == 0, :] = 0
        return rgb

    def update(self, scans: LidarScanSet) -> LidarScanSet:
        for ith_scan, scan in enumerate(scans):
            stacked_result_rgb = np.empty((scan.h * len(self.field_to_util[ith_scan].keys()), scan.w, 3), np.uint8)
            for i, field in enumerate(self.field_to_util[ith_scan].keys()):
                img_mono = destagger(scan.sensor_info, scan.field(field)).astype(np.float32)
                self.field_to_util[ith_scan][field]["ae"].update(img_mono)
                self.field_to_util[ith_scan][field]["buc"].update(img_mono)
                img_rgb = np.repeat(np.uint8(np.clip(np.rint(img_mono * 255), 0, 255))[..., np.newaxis], 3, axis=-1)
                imgsz = [img_rgb.shape[0] * 2, img_rgb.shape[1] * 2]
                results: Results = next(
                    self.field_to_util[ith_scan][field]["model"].track(
                        [img_rgb],
                        stream=True,
                        persist=True,
                        conf=0.1,
                        imgsz=imgsz,
                        classes=self.classes_to_detect,
                        retina_masks=True,
                    )).cpu()
                img_rgb_with_results = results.plot(boxes=True, masks=True, line_width=1, font_size=3)

                if self._use_opencv:
                    stacked_result_rgb[i * scan.h:(i + 1) * scan.h, ...] = img_rgb_with_results
                else:
                    instance_id_img, class_id_img, instance_ids, class_ids = self._create_filled_masks(results, scan)
                    xyz_meters = self._xyzluts[ith_scan](scan.field(ChanField.RANGE))
                    range_mm = scan.field(ChanField.RANGE)
                    xyz_meters = destagger(scan.sensor_info, xyz_meters)
                    range_mm = destagger(scan.sensor_info, range_mm)
                    valid = range_mm != 0

                    for instance_id in instance_ids:
                        data_slice = (instance_id_img == instance_id) & valid
                        xyz_slice = xyz_meters[data_slice, :]
                        range_slice_mm = range_mm[data_slice]
                        print(f"ID {instance_id}: {np.median(range_slice_mm) / 1000:0.2f} m, {np.array2string(np.median(xyz_slice, axis=0), precision=2)} m")

                    scan.add_field(f"INSTANCE_ID_{field}", destagger(scan.sensor_info, instance_id_img, inverse=True))
                    scan.add_field(f"CLASS_ID_{field}", destagger(scan.sensor_info, class_id_img, inverse=True))
                    scan.add_field(f"RGB_INSTANCE_ID_{field}", destagger(scan.sensor_info, self.mono_to_rgb(instance_id_img, img_mono), inverse=True))

            if self._use_opencv:
                cv2.imshow("results", stacked_result_rgb)
                cv2.waitKey(1)
        return scans

    def _create_filled_masks(self, results: Results, scan: LidarScan):
        instance_ids = np.empty(0, np.uint32)
        class_ids = np.empty(0, np.uint32)
        if results.boxes.id is not None and results.masks is not None:
            mask_edges = results.masks.xy
            orig_instance_ids = np.uint32(results.boxes.id.int())
            orig_class_ids = np.uint32(results.boxes.cls.int())

            instance_id_img = np.zeros((scan.h, scan.w, 3), np.float32)

            for edge, instance_id, class_id in zip(mask_edges[::-1], orig_instance_ids[::-1], orig_class_ids[::-1]):
                if len(edge) != 0:
                    instance_id_img = cv2.drawContours(instance_id_img, [np.int32([edge])], -1,
                                                       color=[np.float64(instance_id), 0, 0], thickness=-1)
                    instance_ids = np.append(instance_ids, instance_id)
                    class_ids = np.append(class_ids, class_id)
            instance_id_img = instance_id_img[..., 0].astype(np.uint32)
            in_bool = np.isin(instance_ids, instance_id_img)
            instance_ids = instance_ids[in_bool]
            class_ids = class_ids[in_bool]
        else:
            instance_id_img = np.zeros((scan.h, scan.w), np.uint32)

        if instance_ids.size > 0:
            instance_to_class_lut = np.arange(0, np.max(instance_ids) + 1, dtype=np.uint32)
            instance_to_class_lut[instance_ids] = class_ids
            class_id_img = instance_to_class_lut[instance_id_img]
        else:
            class_id_img = np.zeros((scan.h, scan.w), np.uint32)

        return instance_id_img, class_id_img, instance_ids, class_ids

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='sdk yolo demo')
    parser.add_argument('source', type=str)
    args = parser.parse_args()
    source = SourceIterator(open_source(args.source), use_opencv=True)
    for i, scans in enumerate(source):
        if i > 10:
            break
    source = SourceIterator(open_source(args.source), use_opencv=False)
    SimpleViz(source.sensor_info, rate=0).run(source)
