# Improvements on Bounded Attention: Initial Seed Optimization

> **Omri Atir, Roi Mashiah, Ori Assulin**
> 
> In this work, we try to suggest improvements on the paper "Be Yourself: Bounded Attention for Multi-Subject Text-to-Image Generation (ECCV 2024)". This paper deals with layout guidance in text to image models to create spatialy coherent images as well as control over the number of desired objects. The original paper achieves this in inference mode without the need to retrain the model via optimization on the attention layers and manipulation of the way the losses are calculated. This approach yeilds promising results, although, in some cases, it fails to position the correct amount of objects in the bounding boxes. This work tries to promote object creation inside the bounding boxes by tampering with the initial noise that the diffusion model tries to clean. We tried different methods for this matter and we will present how to run each method.

[GitHub Repository](https://github.com/roi-mashiah/ba-noise-control.git)

## Description  
We built upon the existing Bounded Attention [implementation](https://github.com/omer11a/bounded-attention.git). The main approach is as follows - generate an input image that is meant to promote object generation located at the bounding boxes, perform DDIM inversion, use the inverse latent as a seed for Bounded Attention.
## Setup
First we need to clone the two repositories - [our project](https://github.com/roi-mashiah/ba-noise-control.git) and the [inversion project](https://github.com/garibida/ReNoise-Inversion.git) to one directory.
### Environment
To set up the environment, run:
```
conda create --name bounded-attention python=3.11.4
conda activate bounded-attention
pip install -r requirements.txt
```
Then, run in Python:
```
import nltk
nltk.download('averaged_perceptron_tagger')
```
## Usage
Both methods call the Bounded Attention model with random Normal Gaussian noise as the latent for comparison.
The script that is used to create images is `run_sd.py`. The script's main takes a path to the input json file. The json file should have the following format, where the method is an Enum - `MethodType("GPMethod", "APIMethod")`:
```
{
  "1": {
    "prompt": "A train on top of a surfboard.",
    "boxes": [
      [0.25390625,0.15625,0.751953125,0.46875],
      [0.146484375,0.5078125,0.859375,0.87890625]
    ],
    "references": [[2],[7]],
    "background": "railroad",
    "method": "GPMethod"
  }
}
```
The full path should be assigned to the appropriate variable:
```
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_json_path = r"your/path/to/input/json"
    main(input_json_path)
```

### API Patches
In this method we generate a rough version of the desired image via Google Search API and creating patches. A prerequisite for this method is to use your Google account to generate an API Key and a Search Engine ID as explained [here](https://developers.google.com/custom-search/v1/overview). Once you have them, insert the values in the script `create_img.py`. 

To generate images based on the API method, assign `"method": "APIMethod"` in the input json.
### Gaussian Patches 
In this method we generate Normal distribution latents, where the bounding boxes are filled with gaussian samples of a different variance. 
To generate images based on the Gaussian Patches method, assign `"method": "GPMethod"` in the input json or omit the "method" key as this is the default value.

