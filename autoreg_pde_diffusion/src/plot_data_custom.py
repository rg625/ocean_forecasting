import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plot_color_and_name_mapping import getModelName, getFieldIndex, getColormapAndNorm

plt.rcParams["pdf.fonttype"] = 42  # prevent type3 fonts in matplotlib output files
plt.rcParams["ps.fonttype"] = 42
# Set plotting style
plt.style.use("seaborn-v0_8-whitegrid")

cmap = sns.color_palette("icefire", as_cmap=True)

datasetName = "stable"
modelMinMax = (0, 1)
evalMinMax = (0, 1)
sequenceMinMax = (0, 1)
# sequenceMinMax = (8, 9)
timeSteps = (
    [0, 1, 2, 3, 4]
    if datasetName in ["lowRey", "highRey", "stable", "interp", "extrap"]
    else ([9, 49, 129, 209] if datasetName in ["longer"] else [9, 39, 69, 99])
)
spatialZoom = (
    [[20, 84], [0, 64]]
    # [[0, 128], [0, 64]]
    if datasetName in ["lowRey", "highRey", "stable"]
    else (
        [[6, 70], [0, 64]]
        if datasetName in ["interp", "extrap", "longer"]
        else [[0, 40], [0, 40]]
    )
)
field = "vort"

predictionFolder = (
    "/home/rg625/mnt/ocean_forecasting/koopman_autoencoder/results/stable"
)
outputFolder = "/home/rg625/mnt/ocean_forecasting/autoreg_pde_diffusion/src/results"

models = {
    "EXP": "continous_linear_128_2_exp_processed.npz",  # groundTruth_on_stable_only for incompressible
    r"$\Delta$ t = 0.05s": "continous_linear_128_1_processed.npz",
    r"$\Delta$ t = 0.1s": "continous_linear_128_2_processed.npz",
    r"$\Delta$ t = 0.2s": "continous_linear_128_4_processed.npz",
}


modelNames = []
frameData = []
gtFrames = None  # will store ground-truth frames for MSE

for modelName, modelPath in models.items():
    modelNames += [modelName]
    # if modelPath == "groundTruth_on_stable_only.dict":
    #     groundTruthDict = torch.load(os.path.join(predictionFolder, "groundTruth_on_stable_only.dict"))
    if modelPath == "groundTruth.dict":
        groundTruthDict = torch.load(os.path.join(predictionFolder, "groundTruth.dict"))
        groundTruth = groundTruthDict["data"].unsqueeze(0).unsqueeze(0)
        obsMask = (
            groundTruthDict["obsMask"]
            .unsqueeze(1)
            .unsqueeze(2)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        groundTruth = groundTruth * obsMask  # ignore obstacle area
        print("Original ground truth shape: %s" % (str(list(groundTruth.shape))))
        prediction = groundTruth[
            :,
            :,
            sequenceMinMax[0] : sequenceMinMax[1],
            timeSteps,
            :,
            spatialZoom[0][0] : spatialZoom[0][1],
            spatialZoom[1][0] : spatialZoom[1][1],
        ]
        prediction = torch.squeeze(
            prediction[:, :, :, :, getFieldIndex(datasetName, field)]
        )
        # gtFrames = prediction.permute(0, 2, 1).numpy()
        print("Loaded ground truth with shape: %s" % (str(list(prediction.shape))))

    else:
        fullPath = os.path.join(predictionFolder, modelPath)
        prediction = torch.from_numpy(np.load(fullPath)["arr_0"])
        # prediction = prediction * obsMask
        print(f"prediction: {prediction.shape}")
        prediction = prediction[
            modelMinMax[0] : modelMinMax[1],
            evalMinMax[0] : evalMinMax[1],
            sequenceMinMax[0] : sequenceMinMax[1],
            timeSteps,
            :,
            spatialZoom[0][0] : spatialZoom[0][1],
            spatialZoom[1][0] : spatialZoom[1][1],
        ]
        prediction = torch.squeeze(
            prediction[:, :, :, :, getFieldIndex(datasetName, field)]
        )
        print(
            "Loaded prediction from model %s with shape: %s"
            % (modelName, str(list(prediction.shape)))
        )

    if field == "vort":
        vxDx, vxDy = torch.gradient(prediction[:, 0], dim=[1, 2])
        vyDx, vyDy = torch.gradient(prediction[:, 1], dim=[1, 2])
        prediction = vyDx - vxDy  # curl == vorticity

    frameData += [prediction.permute(0, 2, 1).numpy()]


fig, axs = plt.subplots(
    nrows=len(modelNames),
    ncols=len(timeSteps),
    figsize=(4.5, 6.6),
    dpi=250,
    squeeze=False,
)

# Plot the frames
dt = 0.4

for i in range(len(modelNames)):
    for j in range(len(timeSteps)):
        if i == len(modelNames) - 1:
            physical_time = (timeSteps[j] + 1) * dt
            axs[i, j].set_xlabel(f"$t={physical_time:.1f}s$", fontsize=8)
        if j == 0:
            axs[i, j].set_ylabel(getModelName(modelNames[i]), fontsize=8)
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])
        cmap, norm = getColormapAndNorm(datasetName, field)
        im = axs[i, j].imshow(
            frameData[i][j], interpolation="nearest", cmap=cmap, norm=norm
        )

# Reduce space between rows/columns
fig.subplots_adjust(
    left=0.05,
    right=0.87,  # leave space for colorbar
    top=0.97,
    bottom=0.03,
    hspace=-0.80,  # small vertical space between rows
    wspace=0.02,  # small horizontal space between columns
)

# Make colorbar exactly match the height of all rows
# Get the positions of the first and last axes in the first column
first_ax_pos = axs[0, 0].get_position()
last_ax_pos = axs[-1, 0].get_position()
cbar_ax = fig.add_axes(
    [
        0.88,  # x position
        last_ax_pos.y0,  # bottom aligned with last row
        0.025,  # width
        first_ax_pos.y1 - last_ax_pos.y0,  # height from bottom of last to top of first
    ]
)
fig.colorbar(im, cax=cbar_ax)
cbar_ax.tick_params(labelsize=8)

fig.savefig(
    f"{outputFolder}/data_{datasetName}_{field}_timestep.pdf",
    dpi=250,
    bbox_inches="tight",
)
