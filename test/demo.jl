using Images

# Read images
ref = Images.load("../VSI/Julia Implementation/r0.png")
dist = Images.load("../VSI/Julia Implementation/r1.png")

# Calculate the perceptual quality score (VSI)
@time score = VSI_score(ref, dist)

# Output the score
println("VSI score: ", score)