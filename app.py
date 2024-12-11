import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
from leffa.transform import LeffaTransform
from leffa.model import LeffaModel
from leffa.inference import LeffaInference
from utils.garment_agnostic_mask_predictor import AutoMasker
from utils.densepose_predictor import DensePosePredictor
from utils.utils import resize_and_center

import gradio as gr

# Download checkpoints
snapshot_download(repo_id="franciszzj/Leffa", local_dir="./ckpts")


def leffa_predict(src_image_path, ref_image_path, control_type):
    assert control_type in [
        "virtual_tryon", "pose_transfer"], "Invalid control type: {}".format(control_type)
    src_image = Image.open(src_image_path)
    ref_image = Image.open(ref_image_path)
    src_image = resize_and_center(src_image, 768, 1024)
    ref_image = resize_and_center(ref_image, 768, 1024)

    src_image_array = np.array(src_image)
    ref_image_array = np.array(ref_image)

    # Mask
    if control_type == "virtual_tryon":
        automasker = AutoMasker(
            densepose_path="./ckpts/densepose",
            schp_path="./ckpts/schp",
        )
        src_image = src_image.convert("RGB")
        mask = automasker(src_image, "upper")["mask"]
    elif control_type == "pose_transfer":
        mask = Image.fromarray(np.ones_like(src_image_array) * 255)

    # DensePose
    densepose_predictor = DensePosePredictor(
        config_path="./ckpts/densepose/densepose_rcnn_R_50_FPN_s1x.yaml",
        weights_path="./ckpts/densepose/model_final_162be9.pkl",
    )
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


def leffa_predict_vt(src_image_path, ref_image_path):
    return leffa_predict(src_image_path, ref_image_path, "virtual_tryon")


def leffa_predict_pt(src_image_path, ref_image_path):
    return leffa_predict(src_image_path, ref_image_path, "pose_transfer")


if __name__ == "__main__":
    # import sys

    # src_image_path = sys.argv[1]
    # ref_image_path = sys.argv[2]
    # control_type = sys.argv[3]
    # leffa_predict(src_image_path, ref_image_path, control_type)

    title = "## Leffa: Learning Flow Fields in Attention for Controllable Person Image Generation"
    description = "Leffa is a unified framework for controllable person image generation that enables precise manipulation of both appearance (i.e., virtual try-on) and pose (i.e., pose transfer)."

    with gr.Blocks(theme=gr.themes.Default(primary_hue=gr.themes.colors.pink, secondary_hue=gr.themes.colors.red)).queue() as demo:
        gr.Markdown(title)
        gr.Markdown(description)

        with gr.Tab("Control Appearance (Virtual Try-on)"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Person Image")
                    vt_src_image = gr.Image(
                        sources=["upload"],
                        type="filepath",
                        label="Person Image",
                        width=512,
                        height=512,
                    )

                    gr.Examples(
                        inputs=vt_src_image,
                        examples_per_page=5,
                        examples=["./ckpts/examples/person1/01320_00.jpg",
                                  "./ckpts/examples/person1/01350_00.jpg",
                                  "./ckpts/examples/person1/01365_00.jpg",
                                  "./ckpts/examples/person1/01376_00.jpg",
                                  "./ckpts/examples/person1/01416_00.jpg",],
                    )

                with gr.Column():
                    gr.Markdown("#### Garment Image")
                    vt_ref_image = gr.Image(
                        sources=["upload"],
                        type="filepath",
                        label="Garment Image",
                        width=512,
                        height=512,
                    )

                    gr.Examples(
                        inputs=vt_ref_image,
                        examples_per_page=5,
                        examples=["./ckpts/examples/garment/01449_00.jpg",
                                  "./ckpts/examples/garment/01486_00.jpg",
                                  "./ckpts/examples/garment/01853_00.jpg",
                                  "./ckpts/examples/garment/02070_00.jpg",
                                  "./ckpts/examples/garment/03553_00.jpg",],
                    )

                with gr.Column():
                    gr.Markdown("#### Generated Image")
                    vt_gen_image = gr.Image(
                        label="Generated Image",
                        width=512,
                        height=512,
                    )

                    with gr.Row():
                        vt_gen_button = gr.Button("Generate")

                vt_gen_button.click(fn=leffa_predict_vt, inputs=[
                    vt_src_image, vt_ref_image], outputs=[vt_gen_image])

        with gr.Tab("Control Pose (Pose Transfer)"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Person Image")
                    pt_ref_image = gr.Image(
                        sources=["upload"],
                        type="filepath",
                        label="Person Image",
                        width=512,
                        height=512,
                    )

                    gr.Examples(
                        inputs=vt_src_image,
                        examples_per_page=5,
                        examples=["./ckpts/examples/person1/01320_00.jpg",
                                  "./ckpts/examples/person1/01350_00.jpg",
                                  "./ckpts/examples/person1/01365_00.jpg",
                                  "./ckpts/examples/person1/01376_00.jpg",
                                  "./ckpts/examples/person1/01416_00.jpg",],
                    )

                with gr.Column():
                    gr.Markdown("#### Target Pose Person Image")
                    pt_src_image = gr.Image(
                        sources=["upload"],
                        type="filepath",
                        label="Target Pose Person Image",
                        width=512,
                        height=512,
                    )

                    gr.Examples(
                        inputs=pt_src_image,
                        examples_per_page=5,
                        examples=["./ckpts/examples/person2/01850_00.jpg",
                                  "./ckpts/examples/person2/01875_00.jpg",
                                  "./ckpts/examples/person2/02532_00.jpg",
                                  "./ckpts/examples/person2/02902_00.jpg",
                                  "./ckpts/examples/person2/05346_00.jpg",],
                    )

                with gr.Column():
                    gr.Markdown("#### Generated Image")
                    pt_gen_image = gr.Image(
                        label="Generated Image",
                        width=512,
                        height=512,
                    )

                    with gr.Row():
                        pose_transfer_gen_button = gr.Button("Generate")

                pose_transfer_gen_button.click(fn=leffa_predict_pt, inputs=[
                    pt_src_image, pt_ref_image], outputs=[pt_gen_image])

        demo.launch(share=True, server_port=7860)
