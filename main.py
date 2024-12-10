import numpy as np
from PIL import Image
from leffa.transform import LeffaTransform
from leffa.model import LeffaModel
from leffa.inference import LeffaInference
from utils.garment_agnostic_mask_predictor import AutoMasker
from utils.densepose_predictor import DensePosePredictor


def leffa_predict(src_image_path, ref_image_path):
    src_image = Image.open(src_image_path)
    ref_image = Image.open(ref_image_path)

    src_image_array = np.array(src_image)
    ref_image_array = np.array(ref_image)

    # Mask
    automasker = AutoMasker()
    src_image = src_image.convert("RGB")
    mask = automasker(src_image, "upper")["mask"]

    # DensePose
    densepose_predictor = DensePosePredictor()
    src_image_iuv_array = densepose_predictor.predict_iuv(src_image_array)
    src_image_seg_array = densepose_predictor.predict_seg(src_image_array)
    src_image_iuv = Image.fromarray(src_image_iuv_array)
    src_image_seg = Image.fromarray(src_image_seg_array)

    # Leffa
    transform = LeffaTransform()
    model = LeffaModel(
        pretrained_model_name_or_path="",
        pretrained_garmentnet_path="",
        pretrained_model="",
    )
    inference = LeffaInference(model=model)

    data = {
        "src_image": [src_image],
        "ref_image": [ref_image],
        "mask": [mask],
        "densepose": [src_image_iuv],
    }
    data = transform(data)
    output = inference(data)
    gen_image = output["generated_image"][0]
    gen_image.save("gen_image.png")
