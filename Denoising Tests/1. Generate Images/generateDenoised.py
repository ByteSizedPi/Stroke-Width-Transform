import cv2
import deeplake
import numpy as np
from deeplake import Dataset
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle

ds: Dataset = deeplake.load("hub://activeloop/icdar-2013-text-localize-train")


def perform_filters(cur_img, idx):

    all_filters = {}

    all_filters.update({"original": cur_img})

    print("Original done for index", idx)

    # all_filters.update(
    #     {
    #         f"cvBilateral_sigma_{i}": cv2.bilateralFilter(cur_img, 10, i, i)
    #         for i in range(100, 400, 100)
    #     }
    # )

    # print("cvBilateral done for index", idx)

    # all_filters.update(
    #     {"skimage_bilateral": denoise_bilateral(cur_img, channel_axis=-1)}
    # )

    # print("skimage_bilateral done for index", idx)

    # all_filters.update(
    #     {
    #         f"bilateralTexture_sigma={i}": cv2.ximgproc.bilateralTextureFilter(
    #             cur_img, sigmaAlpha=i, sigmaAvg=i
    #         )
    #         for i in range(-1, 600, 200)
    #     }
    # )

    # print("bilateralTexture done for index", idx)

    # all_filters.update(
    #     {
    #         f"jointBilateral_sigma={i}": cv2.ximgproc.jointBilateralFilter(
    #             cur_img, cur_img, d=5, sigmaColor=i, sigmaSpace=i
    #         )
    #         for i in range(-1, 600, 200)
    #     }
    # )

    # print("jointBilateral done for index", idx)

    # all_filters.update(
    #     {
    #         f"guidedFilter_eps={i}": cv2.ximgproc.guidedFilter(
    #             cur_img, cur_img, radius=10, eps=i
    #         )
    #         for i in range(0, 800, 200)
    #     }
    # )

    # print("guidedFilter done for index", idx)

    # all_filters.update(
    #     {
    #         f"weightedMedian_sigma={i}": cv2.ximgproc.weightedMedianFilter(
    #             cur_img, cur_img, r=10, sigma=i
    #         )
    #         for i in range(1, 800, 200)
    #     }
    # )

    # print("weightedMedian done for index", idx)

    # all_filters.update(
    #     {
    #         f"edgePreserving_threshold={i}": cv2.ximgproc.edgePreservingFilter(
    #             cur_img, d=3, threshold=i
    #         )
    #         for i in range(1, 100, 20)
    #     }
    # )

    # print("edgePreserving done for index", idx)

    # all_filters.update(
    #     {
    #         f"anisotropicDiffusion_alpha={i}": cv2.ximgproc.anisotropicDiffusion(
    #             cur_img, i, 100, 10
    #         )
    #         for i in np.arange(0.04, 0.2, 0.03)
    #     }
    # )

    # print("anisotropicDiffusion done for index", idx)

    # all_filters.update({f"totalVariation": denoise_tv_chambolle(cur_img)})

    # print("totalVariation done for index", idx)

    return all_filters


for i in range(0, 230):
    try:
        print(
            f"---------------------------------Processing index {i}---------------------------------"
        )
        mod_dict = perform_filters(ds.images[i].numpy(), i)
        print(
            f"---------------------------------Done with index {i}---------------------------------"
        )
        for key, value in mod_dict.items():
            cv2.imwrite(f"Denoising Tests/results/{key}_index={i}.png", value)
        print(
            f"---------------------------------Images Saved---------------------------------"
        )
    except:
        print(f"Error at index {i}")
        break
