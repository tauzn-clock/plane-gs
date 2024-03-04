from typing import Dict, ForwardRef, Generic, List, Literal, Optional, Tuple, Type, Union, cast, get_args, get_origin
from dataclasses import dataclass, field
import random

from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig, TDataset
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig, FullImageDatamanager
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils.rich_utils import CONSOLE

@dataclass
class PlaneGSDatamanagerConfig(FullImageDatamanagerConfig):
    _target: Type = field(default_factory=lambda: PlaneGSDatamanager)


class PlaneGSDatamanager(FullImageDatamanager, Generic[TDataset]):
    """
    A datamanager that outputs full images and cameras instead of raybundles. This makes the
    datamanager more lightweight since we don't have to do generate rays. Useful for full-image
    training e.g. rasterization pipelines
    """

    config: PlaneGSDatamanagerConfig

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next training batch

        Returns a Camera instead of raybundle"""
        image_idx = self.train_unseen_cameras.pop(random.randint(0, len(self.train_unseen_cameras) - 1))
        CONSOLE.log(image_idx)
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.train_unseen_cameras) == 0:
            self.train_unseen_cameras = [i for i in range(len(self.train_dataset))]

        assert len(self.train_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.train_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = image_idx
        return camera, self.cached_train[image_idx]