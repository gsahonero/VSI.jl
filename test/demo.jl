using Images
using VSI

# Read images
ref = Images.load("./test/images/r0.png")
dist = Images.load("./test/images/r1.png")

# Calculate the perceptual quality score (VSI)
@time score = VSI_score(ref, ref)

# Output the score
println("VSI score: ", score)