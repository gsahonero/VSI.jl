# VSI

[![Build Status](https://github.com/gsahonero/VSI.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/gsahonero/VSI.jl/actions/workflows/CI.yml?query=branch%3Amaster)

A Julia implementation of Visual Saliency-Induced Index (VSI) metric. This implementation follows the logic implemented in [this blog post](https://imageprocessing-sankarsrin.blogspot.com/2017/10/vsi-visual-saliency-induced-index-image.html). Please, check [comments](#comments) before using.

## Setup
For now: 

```julia
using Pkg;
Pkg.add(url="https://github.com/gsahonero/VSI.jl")
Pkg.dev("VSI")
```

## Usage

The classic example: 

```julia
using VSI
using Images

# Read images
using Images

# Read images
ref = Images.load("../VSI/Julia Implementation/r0.png")
dist = Images.load("../VSI/Julia Implementation/r1.png")

# Calculate the perceptual quality score (VSI)
@time score = VSI_score(ref, dist)

# Output the score
println("VSI score: ", score)
```

## Parameters of `VSI_score`
- `ref`:`Image` - reference image
- `dist`:`Image` - distorted image

## Comments
- (AFAIK) Julia does not have an implementation of the VSI metric, this "package" should do the trick. 
- Due to differences with MATLAB implementations and numerical operations, the results may differ slightly.