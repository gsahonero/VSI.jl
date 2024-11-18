using VSI
using Test
using Images

@testset "VSI.jl" begin
    ref = Images.load(joinpath(pwd(), "images","r0.png"))
    dist = Images.load(joinpath(pwd(), "images","r1.png"))
    @test round(VSI.VSI_score(ref, ref))==1
end
