import os
import time
import sys
import shutil
import copy
import torch
from torch.utils.data import DataLoader, SequentialSampler
import numpy as np

from turbpred.model import PredictionModel
from turbpred.model_diffusion import DiffusionModel
from turbpred.turbulence_dataset import TurbulenceDataset
from turbpred.data_transformations import Transforms
from turbpred.params import (
    DataParams,
    TrainingParams,
    LossParams,
    ModelParamsEncoder,
    ModelParamsDecoder,
    ModelParamsLatent,
)
import json

device = "cuda"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


outFolder = "/home/rg625/mnt/ocean_forecasting/autoreg_pde_diffusion/results/sampling"
modelFolder = "/home/rg625/mnt/ocean_forecasting/autoreg_pde_diffusion/pretrained_models/models_tra"

testSets = {
    "extrap": TurbulenceDataset(
        "Test Extrapolate Mach 0.50-0.52",
        ["/home/rg625/mnt/ocean_forecasting/data"],
        filterTop=["128_tra"],
        filterSim=[(0, 3)],
        filterFrame=[(500, 750)],
        sequenceLength=[[120, 1]],
        simFields=["dens", "pres"],
        simParams=["mach"],
        printLevel="sim",
    ),
    "interp": TurbulenceDataset(
        "Test Interpolate Mach 0.66-0.68",
        ["/home/rg625/mnt/ocean_forecasting/data"],
        filterTop=["128_tra"],
        filterSim=[(16, 19)],
        filterFrame=[(500, 750)],
        sequenceLength=[[120, 1]],
        simFields=["dens", "pres"],
        simParams=["mach"],
        printLevel="sim",
    ),
    "longer": TurbulenceDataset(
        "Test Longer Rollout Mach 0.64-0.65",
        ["/home/rg625/mnt/ocean_forecasting/data"],
        filterTop=["128_tra"],
        filterSim=[(14, 16)],
        filterFrame=[(0, 1000)],
        sequenceLength=[[480, 1]],
        simFields=["dens", "pres"],
        simParams=["mach"],
        printLevel="sim",
    ),
}
dataTransformationFieldIndices = [
    0,
    1,
    2,
    3,
    5,
]  # ORDER (fields): velocity (x,y), velocity z / density, pressure, ORDER (params): rey, mach, zslice

numEvals = 5

models = {
    "tf-enc": (
        [
            "128_tf-enc_00/Model.pth",
            "128_tf-enc_01/Model.pth",
            "128_tf-enc_02/Model.pth",
        ],
        {"extrap": 8, "interp": 8, "longer": 1},
        {
            "numEvals": 1,
        },
    ),
    "tf-vae": (
        [
            "128_tf-vae_00/Model.pth",
            "128_tf-vae_01/Model.pth",
            "128_tf-vae_02/Model.pth",
        ],
        {"extrap": 4, "interp": 4, "longer": 4},
        {
            "numEvals": numEvals,
            "sequentialEvalRuns": {"extrap": False, "interp": False, "longer": True},
        },
    ),
    "tf-mgn": (
        [
            "128_tf-mgn_00/Model.pth",
            "128_tf-mgn_01/Model.pth",
            "128_tf-mgn_02/Model.pth",
        ],
        {"extrap": 8, "interp": 8, "longer": 1},
        {
            "numEvals": 1,
        },
    ),
    "unet-m2": (
        [
            "128_unet-m2_00/Model.pth",
            "128_unet-m2_01/Model.pth",
            "128_unet-m2_02/Model.pth",
        ],
        {"extrap": 8, "interp": 8, "longer": 1},
        {
            "numEvals": 1,
        },
    ),
    "unet-m8": (
        [
            "128_unet-m8_00/Model.pth",
            "128_unet-m8_01/Model.pth",
            "128_unet-m8_02/Model.pth",
        ],
        {"extrap": 8, "interp": 8, "longer": 1},
        {
            "numEvals": 1,
        },
    ),
    "unet-m2-noise0.01": (
        [
            "128_unet-m2-noise0.01_00/Model.pth",
            "128_unet-m2-noise0.01_01/Model.pth",
            "128_unet-m2-noise0.01_02/Model.pth",
        ],
        {"extrap": 8, "interp": 8, "longer": 1},
        {
            "numEvals": 1,
        },
    ),
    "refiner-r4_std0.000001": (
        [
            "128_refiner-r4_std0.000001_00/Model.pth",
            "128_refiner-r4_std0.000001_01/Model.pth",
            "128_refiner-r4_std0.000001_02/Model.pth",
        ],
        {"extrap": 8, "interp": 8, "longer": 4},
        {
            "numEvals": numEvals,
            "sequentialEvalRuns": {"extrap": True, "interp": True, "longer": True},
        },
    ),
    "resnet-m2": (
        [
            "128_resnet-m2_00/Model.pth",
            "128_resnet-m2_01/Model.pth",
            "128_resnet-m2_02/Model.pth",
        ],
        {"extrap": 8, "interp": 8, "longer": 1},
        {
            "numEvals": 1,
        },
    ),
    "dil-resnet-m2": (
        [
            "128_dil-resnet-m2_00/Model.pth",
            "128_dil-resnet-m2_01/Model.pth",
            "128_dil-resnet-m2_02/Model.pth",
        ],
        {"extrap": 8, "interp": 8, "longer": 1},
        {
            "numEvals": 1,
        },
    ),
    "fno-16modes-m2": (
        [
            "128_fno-16modes-m2_00/Model.pth",
            "128_fno-16modes-m2_01/Model.pth",
            "128_fno-16modes-m2_02/Model.pth",
        ],
        {"extrap": 8, "interp": 8, "longer": 1},
        {
            "numEvals": 1,
        },
    ),
    "fno-32modes-m2": (
        [
            "128_fno-32modes-m2_00/Model.pth",
            "128_fno-32modes-m2_01/Model.pth",
            "128_fno-32modes-m2_02/Model.pth",
        ],
        {"extrap": 8, "interp": 8, "longer": 1},
        {
            "numEvals": 1,
        },
    ),
    "acdm-r20_ncn": (
        [
            "128_acdm-r20_ncn_00/Model.pth",
            "128_acdm-r20_ncn_01/Model.pth",
            "128_acdm-r20_ncn_02/Model.pth",
        ],
        {"extrap": 8, "interp": 8, "longer": 4},
        {
            "numEvals": numEvals,
            "sequentialEvalRuns": {"extrap": True, "interp": True, "longer": True},
            "samplingMode": "ddpm",
            "posteriorSampling": "random",
            "initialSampling": "random",
            "conditioningIntegration": "clean",
        },
    ),
    "acdm-r20": (
        [
            "128_acdm-r20_00/Model.pth",
            "128_acdm-r20_01/Model.pth",
            "128_acdm-r20_02/Model.pth",
        ],
        {"extrap": 8, "interp": 8, "longer": 4},
        {
            "numEvals": numEvals,
            "sequentialEvalRuns": {"extrap": True, "interp": True, "longer": True},
            "samplingMode": "ddpm",
            "posteriorSampling": "random",
            "initialSampling": "random",
            "conditioningIntegration": "noisy",
        },
    ),
    "uncond-ddpm": (
        ["/home/rg625/mnt/ocean_forecasting/autoreg_pde_diffusion/src/runs/2D_Tra/128_uncond-ddpm_01/Model.pth"],
        {"extrap": 4, "interp": 4, "longer": 4},
        {
            "numEvals": 1,
            "sequentialEvalRuns": {"extrap": True, "interp": True, "longer": True},
            "samplingMode": "ddpm",
            "posteriorSampling": "random",
            "initialSampling": "random",
            "conditioningIntegration": "noisy",
        },
    ),
}

# Sample ONLY the unconditional model on this run (delete this line to sample all above;
# models whose .npz already exists in results/sampling/<regime> are skipped anyway).
models = {"uncond-ddpm": models["uncond-ddpm"]}

os.makedirs(outFolder, exist_ok=True)


def measure_inference_time_and_vram(model, data, simParameters):
    """
    Returns:
        prediction: model output
        elapsed_ms: average ms per sequence
        max_vram_MB: max VRAM (MB) used during this forward pass
    """
    num_sequences = data.shape[0]

    torch.cuda.synchronize()
    torch.cuda.reset_max_memory_allocated(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    with torch.no_grad():
        prediction, _, _ = model(data, simParameters)
    end_event.record()

    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)

    # Max VRAM in MB
    max_vram_MB = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    # Return normalized time (ms per sequence)
    return prediction, (elapsed_ms / num_sequences), max_vram_MB

# -----------------------------------------
# Save ground truth for test sets
# -----------------------------------------
for testName, testSet in testSets.items():
    testSetOutPath = os.path.join(outFolder, testName)
    os.makedirs(testSetOutPath, exist_ok=True)

    shutil.copy(sys.argv[0], os.path.join(testSetOutPath, "benchmark_script.py"))

    if os.path.isfile(os.path.join(testSetOutPath, "groundTruth_subsample_1.dict")):
        print(f"Skipping {testSetOutPath}/groundTruth_subsample_1.dict")
        continue

    testSet.transform = lambda x: x
    testLoader = DataLoader(testSet, sampler=SequentialSampler(testSet), batch_size=len(testSet), drop_last=False)
    sample = next(iter(testLoader))

    torch.save(sample, os.path.join(testSetOutPath, "groundTruth_subsample_1.dict"))
    print(f"Saved ground truth to {testSetOutPath}/groundTruth_subsample_1.dict")

# -----------------------------------------
# Run model inference, timing, and VRAM
# -----------------------------------------
for modelName, modelData in models.items():
    print(f"\n\n{'-'*50}\nProcessing model: {modelName}\n{'-'*50}")

    modelPaths, batchDict, evalOptions = modelData
    all_step_times = []
    all_vram_MB = []

    for testName, testSet in testSets.items():
        testSetOutPath = os.path.join(outFolder, testName)
        os.makedirs(testSetOutPath, exist_ok=True)
        if os.path.isfile(os.path.join(testSetOutPath, modelName + ".npz")):
            print(f"Skipping {testSetOutPath}/{modelName}.npz")
            continue

        predFull = []
        timerFullStart = time.perf_counter()

        for modelPath in modelPaths:
            model = PredictionModel.load(os.path.join(modelFolder, modelPath), useGPU=device == "cuda")
            model.eval()

            # Update diffusion sampling options
            if isinstance(model.modelDecoder, DiffusionModel):
                for opt in ["samplingMode", "posteriorSampling", "initialSampling", "conditioningIntegration"]:
                    if opt in evalOptions:
                        setattr(model.modelDecoder, f"inference{opt[0].upper() + opt[1:]}", evalOptions[opt])
            elif isinstance(model.modelDecoder, torch.nn.ModuleList):
                for module in model.modelDecoder:
                    if isinstance(module, DiffusionModel):
                        for opt in ["samplingMode", "posteriorSampling", "initialSampling", "conditioningIntegration"]:
                            if opt in evalOptions:
                                setattr(module, f"inference{opt[0].upper() + opt[1:]}", evalOptions[opt])

            batchSize = batchDict[testName]

            # Prepare test transformations
            p_d_test = copy.deepcopy(model.p_d)
            p_d_test.augmentations = ["normalize"]
            p_d_test.sequenceLength = testSet.sequenceLength
            p_d_test.randSeqOffset = False
            testTransformations = Transforms(p_d_test)
            testSet.transform = testTransformations

            testLoader = DataLoader(testSet, sampler=SequentialSampler(testSet), batch_size=batchSize, drop_last=False)

            with torch.no_grad():
                predEvals = []
                numEvalsOption = evalOptions["numEvals"]

                sequential = numEvalsOption == 1 or evalOptions.get("sequentialEvalRuns", {}).get(testName, False)
                print(f"\tSampling {'sequentially' if sequential else 'in parallel'} for {numEvalsOption} samples...")

                timerEvalsStart = time.perf_counter()
                for run in range(numEvalsOption if sequential else 1):
                    predSamples = []
                    for sample in testLoader:
                        data = sample["data"].to(device)
                        simParameters = sample["simParameters"].to(device) if not isinstance(sample["simParameters"], dict) else None
                        if not sequential:
                            # Repeat batch for parallel eval
                            data = data.repeat(numEvalsOption, 1, 1, 1, 1)
                            if simParameters is not None:
                                simParameters = simParameters.repeat(numEvalsOption, 1, 1)

                        prediction, step_time, vram_MB = measure_inference_time_and_vram(model, data, simParameters)
                        all_step_times.append(step_time)
                        all_vram_MB.append(vram_MB)

                        prediction = prediction.unsqueeze(0).unsqueeze(0)
                        predSamples.append(prediction.cpu().numpy())

                    predSamples = np.concatenate(predSamples, axis=2)
                    if sequential and numEvalsOption == 1:
                        predSamples = np.repeat(predSamples, numEvals, axis=1)

                    # Undo normalization
                    normMean = np.expand_dims(testTransformations.normMean[dataTransformationFieldIndices], axis=(0,1,2,3,5,6))
                    normStd = np.expand_dims(testTransformations.normStd[dataTransformationFieldIndices], axis=(0,1,2,3,5,6))
                    predSamples = (predSamples * normStd) + normMean

                    predEvals.append(predSamples)

                predEvals = np.concatenate(predEvals, axis=1)
                predFull.append(predEvals)

        predFull = np.concatenate(predFull, axis=0)
        np.savez_compressed(os.path.join(testSetOutPath, modelName + ".npz"), predFull)

        # Save timing & VRAM stats
        timing_file = os.path.join(outFolder, f"{modelName}_timings.json")
        timing_results = {
            "total_time_ms": float(np.sum(all_step_times)),
            "average_step_time_ms": float(np.mean(all_step_times)),
            "all_step_times_ms": [float(x) for x in all_step_times],
            "max_vram_MB": float(np.max(all_vram_MB)),
            "mean_vram_MB": float(np.mean(all_vram_MB)),
            "all_vram_MB": [float(x) for x in all_vram_MB]
        }

        with open(timing_file, "w") as f:
            json.dump({modelName: timing_results}, f, indent=4)

        timerFullEnd = time.perf_counter()
        print(f"\n[{(timerFullEnd - timerFullStart)/60:.2f} min] Saved prediction to {testSetOutPath}/{modelName}.npz with shape {predFull.shape}")
        print(f"\tMax VRAM used: {timing_results['max_vram_MB']:.2f} MB, Avg per step: {timing_results['mean_vram_MB']:.2f} MB")

print("\n\nSampling complete.")
