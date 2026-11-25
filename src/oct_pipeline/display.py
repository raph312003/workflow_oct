import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import cv2


class BinaryClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if self.category == 1:
            return model_output.squeeze()

        return -model_output.squeeze()

def bounding_boxs_lesions(labels, nombre_lesions):
    mask = np.zeros_like(labels, dtype=np.uint16)
    bmask = np.zeros_like(labels, dtype=np.uint8)

    for idx, i in enumerate(range(len(nombre_lesions))):
        lesion = labels == idx + 1 # 1,2,3,...

        ys, xs = np.where(lesion>0)

        if len(xs) > 0:
            x_min = xs.min()
            y_min = ys.min()
            x_max = xs.max()
            y_max = ys.max()

        cote_x = x_max - x_min +1
        cote_y = y_max - y_min +1

        ratio_x = np.random.uniform(0.10,0.50)
        ratio_y = np.random.uniform(0.10,0.50)


        ajout_x = int(cote_x*ratio_x)  # cote_x + 0-50%
        ajout_y = int(cote_y*ratio_y) # cote_y + 0-50%

        bas = np.random.randint(0, ajout_y+1)
        haut = ajout_y - bas

        gauche = np.random.randint(0, ajout_x+1)
        droite = ajout_x - gauche

        # Nouvelles coordonnées de la bounding box
        bx_min = max(0, x_min - gauche)
        by_min = max(0, y_min - haut)

        bx_max = min(lesion.shape[1], x_max + droite + 1)
        by_max = min(lesion.shape[0], y_max + bas + 1)

        # Boîte précise
        mask[y_min:y_max+1, x_min:x_max+1] = idx + 1

        # Boîte imprécise
        bmask[by_min:by_max, bx_min:bx_max] = idx + 1

        print("idx", idx + 1, "x =",bx_min,"y =",by_min, "area =",bx_max*by_max)
        
    return mask, bmask
    

def show_oct_cam(cube, cam, patient_dir, id_p, id_v, ps):
    
    n = cube[:,0,:,:].numpy().shape[0]
    idx = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    idx = sorted(set(idx))

    # ps = pixel_spacing(ds)
    asp = (ps[0] / ps[1]) if ps else "auto"

    fig, axes = plt.subplots(
        len(idx), 4,
        figsize=(12, 4 * len(idx))
    )

    for ax, i in zip(np.atleast_1d(axes), idx):

        visualizations_0 = []
        visualizations_1 = []

        for i in range(cube.shape[0]):

            bscan = cube[i:i+1]  # [1, 1, H, W]

            with torch.enable_grad():
                grayscale_cam_0 = cam(input_tensor=bscan, targets=[BinaryClassifierOutputTarget(0)])[0]
                grayscale_cam_1 = cam(input_tensor=bscan, targets=[BinaryClassifierOutputTarget(1)])[0]

            bscan_np = bscan[0, 0].detach().cpu().numpy()

            bscan_rgb = np.repeat(
                bscan_np[..., None],
                3,
                axis=-1
            )

            visualization_0 = show_cam_on_image(
                bscan_rgb.astype(np.float32),
                grayscale_cam_0,
                use_rgb=True
            )

            visualization_1 = show_cam_on_image(
                bscan_rgb.astype(np.float32),
                grayscale_cam_1,
                use_rgb=True
            )

            visualizations_0.append(visualization_0)
            visualizations_1.append(visualization_1)

        arr_vis_0 = np.array(visualizations_0)
        arr_vis_1 = np.array(visualizations_1)

        ax[0].imshow(
            arr_vis_0[i], # autoscale(cube[i])
            cmap="gray",
            aspect=asp
        )
        ax[0].set_title(f"B-scan (y=0) {i}/{n-1}")
        ax[0].axis("off")

        ax[1].imshow(
            arr_vis_1[i], # autoscale(cube[i])
            cmap="gray",
            aspect=asp
        )
        ax[1].set_title(f"B-scan (y=1) {i}/{n-1}")
        ax[1].axis("off")

        threshold = 0.7

        mask = (grayscale_cam_1 >= threshold).astype(np.uint8)

        # contours
        contours, hierarchy = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # print(len(contours))

        # Identifiant chaque contours
        labeled_mask = np.zeros(mask.shape, dtype=np.uint16)

        for i, contour in enumerate(contours, start=1):

            cv2.drawContours(
                labeled_mask,
                [contour],
                contourIdx=-1,
                color=int(i),
                thickness=cv2.FILLED
            )

        # print("Nombre de régions :", len(contours))

        _, bmask = bounding_boxs_lesions(labeled_mask, contours)

        # Conversion CAM into mask
        ax[2].imshow(
            mask, # autoscale(cube[i])
            cmap="gray",
            aspect=asp
        )
        ax[2].set_title(f"B-scan (y=1) {i}/{n-1}")
        ax[2].axis("off")

        # Bounding boxes for different lesion
        ax[3].imshow(
            bmask, # autoscale(cube[i])
            cmap="nipy_spectral",
            aspect=asp
        )
        ax[3].set_title(f"B-scan (y=1) {i}/{n-1}")
        ax[3].axis("off")

    fig.suptitle(
        # f"OPT - {ds.get('ImageLaterality', '?')} "
        f"- {cube.shape} "
        f"range [{cube.min():.0f}, {cube.max():.0f}] "
        f"spacing={ps} "
    )

    fig.savefig(
        os.path.join(patient_dir, f"OPT_cam_{id_p}_{id_v}_.png"),
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(fig)