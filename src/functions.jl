using Images, ImageFiltering, LinearAlgebra
using FFTW, DSP

function rgb2matrix(image)
    ref_img = permutedims(channelview(image),[2,3,1])[:, :, 1:3]
    ref_img = Float32.(ref_img)
    return ref_img
end

function VSI_score(image1, image2)
    constForVS = 1.27
    constForGM = 386
    constForChrom = 130
    alpha = 0.40
    lambda = 0.020
    sigmaF = 1.34
    omega0 = 0.0210
    sigmaD = 145
    sigmaC = 0.001
    
    # Resize to 256x256
    image1 = imresize(image1, (256, 256))
    image2 = imresize(image2, (256, 256))
    image1=floor.(rgb2matrix(image1).*256)
    image2=floor.(rgb2matrix(image2).*256)
    # Compute the visual saliency map using SDSP
    saliencyMap1 = SDSP(image1, sigmaF, omega0, sigmaD, sigmaC)
    saliencyMap2 = SDSP(image2, sigmaF, omega0, sigmaD, sigmaC)

    rows, cols = size(image1)[1:2]
    
    # Transform into opponent color space
    L1 = 0.06 * float(image1[:,:,1]) + 0.63 * float(image1[:,:,2]) + 0.27 * float(image1[:,:,3])
    L2 = 0.06 * float(image2[:,:,1]) + 0.63 * float(image2[:,:,2]) + 0.27 * float(image2[:,:,3])
    M1 = 0.30 * float(image1[:,:,1]) + 0.04 * float(image1[:,:,2]) - 0.35 * float(image1[:,:,3])
    M2 = 0.30 * float(image2[:,:,1]) + 0.04 * float(image2[:,:,2]) - 0.35 * float(image2[:,:,3])
    N1 = 0.34 * float(image1[:,:,1]) - 0.60 * float(image1[:,:,2]) + 0.17 * float(image1[:,:,3])
    N2 = 0.34 * float(image2[:,:,1]) - 0.60 * float(image2[:,:,2]) + 0.17 * float(image2[:,:,3])


    # Downsample the image
    minDimension = min(rows, cols)
    F = max(1, round(minDimension / 256))
    aveKernel = ones(Int(F), Int(F)) / (F^2)  # Approximation for average kernel
    F = Int(F)
    aveM1 = conv(M1, aveKernel)
    aveM2 = conv(M2, aveKernel)
    M1 = aveM1[1:F:rows, 1:F:cols]
    M2 = aveM2[1:F:rows, 1:F:cols]

    aveN1 = conv(N1, aveKernel)
    aveN2 = conv(N2, aveKernel)
    N1 = aveN1[1:F:rows, 1:F:cols]
    N2 = aveN2[1:F:rows, 1:F:cols]

    aveL1 = conv(L1, aveKernel)
    aveL2 = conv(L2, aveKernel)
    L1 = aveL1[1:F:rows, 1:F:cols]
    L2 = aveL2[1:F:rows, 1:F:cols]

    aveSM1 = conv(saliencyMap1, aveKernel)
    aveSM2 = conv(saliencyMap2, aveKernel)
    saliencyMap1 = aveSM1[1:F:rows, 1:F:cols]
    saliencyMap2 = aveSM2[1:F:rows, 1:F:cols]

    # Calculate the gradient map
    dx = [3 0 -3; 10 0 -10; 3 0 -3] / 16
    dy = [3 10 3; 0 0 0; -3 -10 -3] / 16

    #L1 = hcat(zeros(258, 2), vcat(zeros(2, 256), L1))
    IxL1 = conv(L1, dx)
    IxL1 = IxL1[2:end-1,2:end-1]
    IyL1 = conv(L1, dy)
    IyL1 = IyL1[2:end-1, 2:end-1]
    gradientMap1 = sqrt.(IxL1.^2 + IyL1.^2)

    IxL2 = conv(L2, dx)
    IxL2 = IxL2[2:end-1,2:end-1]
    IyL2 = conv(L2, dy)
    IyL2 = IyL2[2:end-1,2:end-1]
    gradientMap2 = sqrt.(IxL2.^2 + IyL2.^2)

    # Calculate the VSI
    VSSimMatrix = (2 * saliencyMap1 .* saliencyMap2 .+ constForVS) ./ (saliencyMap1.^2 .+ saliencyMap2.^2 .+ constForVS)
    gradientSimMatrix = (2 * gradientMap1 .* gradientMap2 .+ constForGM) ./ (gradientMap1.^2 .+ gradientMap2.^2 .+ constForGM)

    weight = max.(saliencyMap1, saliencyMap2)

    ISimMatrix = (2 * M1 .* M2 .+ constForChrom) ./ (M1.^2 + M2.^2 .+ constForChrom)
    QSimMatrix = (2 * N1 .* N2 .+ constForChrom) ./ (N1.^2 + N2.^2 .+ constForChrom)

    SimMatrixC = (gradientSimMatrix .^ alpha) .* (VSSimMatrix .* real((ISimMatrix .* QSimMatrix .+ 0im) .^ lambda) .* weight)
    sim = sum(SimMatrixC) / sum(weight)

    return sim
end

function SDSP(image, sigmaF, omega0, sigmaD, sigmaC)
    # Convert the image to double
    dsImage = float(image)
    oriRows, oriCols = size(image)[1:2]
    
    # Convert RGB to Lab
    lab = RGB2Lab(dsImage)
    LChannel, AChannel, BChannel = lab[:,:,1], lab[:,:,2], lab[:,:,3]

    # Compute FFT of each channel
    LFFT = fft(LChannel)
    AFFT = fft(AChannel)
    BFFT = fft(BChannel)

    # Log-Gabor filter
    LG = logGabor(oriRows, oriCols, omega0, sigmaF)
    
    FinalLResult = real.(ifft(LFFT .* LG))
    FinalAResult = real.(ifft(AFFT .* LG))
    FinalBResult = real.(ifft(BFFT .* LG))
    
    SFMap = sqrt.(FinalLResult.^2 + FinalAResult.^2 + FinalBResult.^2)

    # Center bias map    
    centerY, centerX = oriRows / 2, oriCols / 2
    coordinateMtx = zeros(oriCols, oriRows, 2)
    coordinateMtx[:,:,1] = repeat(1:oriRows, 1, oriCols)' .-centerY
    coordinateMtx[:,:,2] = repeat(1:oriCols, 1, oriRows) .-centerX
    
    SDMap = exp.(-sum((coordinateMtx).^2, dims=3) / sigmaD^2)

    # Warm colors bias
    normalizedA = (AChannel .- minimum(AChannel)) / (maximum(AChannel) - minimum(AChannel))
    normalizedB = (BChannel .- minimum(BChannel)) / (maximum(BChannel) - minimum(BChannel))
    labDistSquare = normalizedA.^2 + normalizedB.^2
    SCMap = 1 .- exp.(-labDistSquare / (sigmaC^2))

    # Final saliency map
    VSMap = SFMap .* SDMap .* SCMap
    VSMap = imresize(VSMap, (oriRows, oriCols))
    VSMap = (VSMap .- minimum(VSMap))./(maximum(VSMap)-minimum(VSMap)) 
    return VSMap
end

function RGB2Lab(image)
    image = float(image)

    # Normalize RGB values to the range [0, 1]
    normalizedR = image[:,:,1] / 255
    normalizedG = image[:,:,2] / 255
    normalizedB = image[:,:,3] / 255

    # Apply the nonlinear transformation (gamma correction)
    RSmallerOrEqualto4045 = normalizedR .<= 0.04045
    RGreaterThan4045 = .!RSmallerOrEqualto4045
    tmpR = (normalizedR / 12.92) .* RSmallerOrEqualto4045
    tmpR .+= (((normalizedR .+ 0.055) / 1.055).^2.4) .* RGreaterThan4045

    GSmallerOrEqualto4045 = normalizedG .<= 0.04045
    GGreaterThan4045 = .!GSmallerOrEqualto4045
    tmpG = (normalizedG / 12.92) .* GSmallerOrEqualto4045
    tmpG .+= (((normalizedG .+ 0.055) / 1.055).^2.4) .* GGreaterThan4045

    BSmallerOrEqualto4045 = normalizedB .<= 0.04045
    BGreaterThan4045 = .!BSmallerOrEqualto4045
    tmpB = (normalizedB / 12.92) .* BSmallerOrEqualto4045
    tmpB .+= (((normalizedB .+ 0.055) / 1.055).^ 2.4) .* BGreaterThan4045

    # Convert RGB to XYZ color space using standard RGB-to-XYZ matrix
    X = tmpR * 0.4124564 + tmpG * 0.3575761 + tmpB * 0.1804375
    Y = tmpR * 0.2126729 + tmpG * 0.7151522 + tmpB * 0.0721750
    Z = tmpR * 0.0193339 + tmpG * 0.1191920 + tmpB * 0.9503041

    # Define the reference white point (D65)
    Xr, Yr, Zr = 0.9642, 1.0, 0.8251  # CIE 1931 D65
    epsilon = 0.008856  # CIE standard
    kappa = 903.3  # CIE standard

    # Normalize XYZ values
    xr = X / Xr
    yr = Y / Yr
    zr = Z / Zr

    # Calculate the nonlinear transformation
    fx = (xr .> epsilon) .* (xr .^ (1 / 3)) .+ (xr .<= epsilon) .* ((kappa * xr .+ 16) / 116)
    fy = (yr .> epsilon) .* (yr .^ (1 / 3)) .+ (yr .<= epsilon) .* ((kappa * yr .+ 16) / 116)
    fz = (zr .> epsilon) .* (zr .^ (1 / 3)) .+ (zr .<= epsilon) .* ((kappa * zr .+ 16) / 116)

    # Calculate Lab values
    labImage = zeros(size(image))  # Create an empty array for the Lab image
    labImage[:,:,1] = 116.0 * fy .- 16.0  # L channel
    labImage[:,:,2] = 500.0 * (fx .- fy)  # a channel
    labImage[:,:,3] = 200.0 * (fy .- fz)  # b channel
    
    return labImage
end

function logGabor(rows, cols, omega0, sigmaF)
    # Generate the u1 and u2 grids (frequency domain coordinates)
    U1, U2 = meshgrid(((1:cols) .- (floor(Int, cols / 2) + 1)) ./ (cols - (cols % 2)), 
    ((1:rows) .- (floor(Int, rows / 2) + 1)) ./ (rows - (rows % 2)))

    # Create the mask to block the high-frequency areas
    mask = ones(rows, cols)
    for rowIndex in 1:rows
        for colIndex in 1:cols
            if (U1[rowIndex, colIndex].^2 + U2[rowIndex, colIndex].^2) .> 0.25
                mask[rowIndex, colIndex] = 0
            end
        end
    end

    # Apply mask to the frequency components
    U1 .= U1 .* mask
    U2 .= U2 .* mask

    # Apply ifftshift to shift the zero frequency component to the center
    U1 = ifftshift(U1)
    U2 = ifftshift(U2)

    # Calculate the radial frequency (radius)
    radius = sqrt.(U1.^2 .+ U2.^2)
    radius[1, 1] = 1  # Avoid division by zero

    # Apply the log-Gabor filter formula
    LG = exp.(-(log.(radius ./ omega0)).^2 ./ (2 * sigmaF^2))
    LG[1, 1] = 0  # Set DC component to 0 (avoid division by zero)

    return LG
end

# Helper function to create a meshgrid in Julia
function meshgrid(xin,yin)
    nx=length(xin)
    ny=length(yin)
    xout=zeros(ny,nx)
    yout=zeros(ny,nx)
    for jx=1:nx
        for ix=1:ny
            xout[ix,jx]=xin[jx]
            yout[ix,jx]=yin[ix]
        end
    end
    return (x=xout, y=yout)
end

export VSI_score