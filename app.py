import numpy as np
from PIL import Image
from leffa.transform import LeffaTransform
from leffa.model import LeffaModel
from leffa.inference import LeffaInference
from utils.garment_agnostic_mask_predictor import AutoMasker
from utils.densepose_predictor import DensePosePredictor

import gradio as gr


def leffa_predict(src_image_path, ref_image_path, control_type):
    assert control_type in [
        "virtual_tryon", "pose_transfer"], "Invalid control type: {}".format(control_type)
    src_image = Image.open(src_image_path)
    ref_image = Image.open(ref_image_path)

    src_image_array = np.array(src_image)
    ref_image_array = np.array(ref_image)

    # Mask
    if control_type == "virtual_tryon":
        automasker = AutoMasker()
        src_image = src_image.convert("RGB")
        mask = automasker(src_image, "upper")["mask"]
    elif control_type == "pose_transfer":
        mask = Image.fromarray(np.ones_like(src_image_array) * 255)

    # DensePose
    densepose_predictor = DensePosePredictor()
    src_image_iuv_array = densepose_predictor.predict_iuv(src_image_array)
    src_image_seg_array = densepose_predictor.predict_seg(src_image_array)
    src_image_iuv = Image.fromarray(src_image_iuv_array)
    src_image_seg = Image.fromarray(src_image_seg_array)
    if control_type == "virtual_tryon":
        densepose = src_image_seg
    elif control_type == "pose_transfer":
        densepose = src_image_iuv

    # Leffa
    transform = LeffaTransform()
    if control_type == "virtual_tryon":
        pretrained_model_name_or_path = "./ckpts/stable-diffusion-inpainting"
        pretrained_model = "./ckpts/virtual_tryon.pth"
    elif control_type == "pose_transfer":
        pretrained_model_name_or_path = "./ckpts/stable-diffusion-xl-1.0-inpainting-0.1"
        pretrained_model = "./ckpts/pose_transfer.pth"
    model = LeffaModel(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        pretrained_model=pretrained_model,
    )
    inference = LeffaInference(model=model)

    data = {
        "src_image": [src_image],
        "ref_image": [ref_image],
        "mask": [mask],
        "densepose": [densepose],
    }
    data = transform(data)
    output = inference(data)
    gen_image = output["generated_image"][0]
    # gen_image.save("gen_image.png")
    return np.array(gen_image)


if __name__ == "__main__":
    # import sys

    # src_image_path = sys.argv[1]
    # ref_image_path = sys.argv[2]
    # control_type = sys.argv[3]
    # leffa_predict(src_image_path, ref_image_path, control_type)

    # Launch a gr.Interface
    gr_demo = gr.Interface(
        fn=leffa_predict,
        inputs=[
            gr.Image(sources=["upload", "webcam", "clipboard"],
                     type="filepath",
                     label="Source Person Image",
                     width=768,
                     height=1024,
                     ),
            gr.Image(sources=["upload", "webcam", "clipboard"],
                     type="filepath",
                     label="Reference Image",
                     width=768,
                     height=1024,
                     ),
            gr.Radio(["virtual_tryon", "pose_transfer"],
                     label="Control Type",
                     default="virtual_tryon",
                     ),
        ],
        outputs=[
            gr.Image(label="Generated Person Image",
                     width=768,
                     height=1024,
                     )
        ],
        title="Leffa",
        description="Leffa is a unified framework for controllable person image generation that enables precise manipulation of both appearance (i.e., virtual try-on) and pose (i.e., pose transfer).",
        article="Controllable person image generation aims to generate a person image conditioned on reference images, allowing precise control over the person’s appearance or pose. However, prior methods often distort fine-grained textural details from the reference image, despite achieving high overall image quality. We attribute these distortions to inadequate attention to corresponding regions in the reference image. To address this, we thereby propose \textbf{learning flow fields in attention} (\textbf{\ours{}}), which explicitly guides the target query to attend to the correct reference key in the attention layer during training. Specifically, it is realized via a regularization loss on top of the attention map within a diffusion-based baseline. Our extensive experiments show that Leffa achieves state-of-the-art performance in controlling appearance (virtual try-on) and pose (pose transfer), significantly reducing fine-grained detail distortion while maintaining high image quality. Additionally, we show that our loss is model-agnostic and can be used to improve the performance of other diffusion models.",
        examples=[
            ["./examples/14092_00_person.jpg", "./examples/04181_00_garment.jpg", "virtual_tryon"],
            ["./examples/14092_00_person.jpg", "./examples/14684_00_person.jpg", "pose_transfer"],
        ],
        # cache_examples=True,
        examples_per_page=10,
        allow_flagging=False,
        theme=gr.themes.Default(),
    )
    gr_demo.launch(share=True, server_port=7860)
