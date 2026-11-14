"""IP-Adapter Plus preprocessor for SD1.5.

The SD1.5 IP-Adapter encoders all read the CLIP ViT-H/14 image tower, which the
legacy table exposes only as ``ip-adapter_clip_sd15`` (label ``ip-adapter_clip_h``).
That label does not name the plus models, so searching the Preprocessor dropdown for
the SD1.5 plus models found nothing. This entry registers the same encoder under an
explicit SD1.5-plus name instead of renaming or re-pointing the legacy entry, so
existing workflows that use ``ip-adapter_clip_h`` keep the identical behaviour.

The encode function and its model cache are the legacy ones (``clip`` /
``unload_clip`` with ``config="clip_h"``); nothing is re-implemented here.
"""

from scripts.preprocessor.legacy.processor import clip, unload_clip
from scripts.supported_preprocessor import Preprocessor, PreprocessorParameter


class PreprocessorIPAdapterSD15Plus(Preprocessor):
    """IP-Adapter Plus for SD1.5 (CLIP ViT-H/14 image encoder)."""

    def __init__(self):
        super().__init__(name="ip-adapter_clip_sd15_plus")
        self.tags = ["IP-Adapter"]
        self.slider_resolution = PreprocessorParameter(visible=False)
        self.slider_1 = PreprocessorParameter(visible=False)
        self.slider_2 = PreprocessorParameter(visible=False)
        self.slider_3 = PreprocessorParameter(visible=False)
        self.returns_image = False
        self.accepts_mask = True
        self.show_control_mode = False
        self.do_not_need_model = False
        self.sorting_priority = 99

    def __call__(
        self,
        input_image,
        resolution=512,
        slider_1=None,
        slider_2=None,
        slider_3=None,
        **kwargs,
    ):
        result, _ = clip(img=input_image, res=resolution, config="clip_h", **kwargs)
        return result

    def unload(self) -> bool:
        unload_clip(config="clip_h")
        return True


Preprocessor.add_supported_preprocessor(PreprocessorIPAdapterSD15Plus())
