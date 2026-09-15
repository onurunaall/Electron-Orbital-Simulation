# Hydrogen Orbital Visualizer, CUDA port

Headless CUDA/C++ reimplementation of the two 3D programs in
[kavan010/Atoms](https://github.com/kavan010/Atoms):

| Original                | This port         | What it does                                                        |
|-------------------------|-------------------|---------------------------------------------------------------------|
| `atom_realtime.cpp`     | `atom_realtime`   | Particles rasterized as small depth-tested spheres over a ground grid |
| `atom_raytracer.cpp`    | `atom_raytracer`  | Particles ray traced with a point light and hard shadows             |

Both programs sample particle positions from |ψ<sub>nlm</sub>|², color them by
probability density, rotate them along the probability current each frame, and
write every frame as a PPM image. There is no window and no mouse/keyboard
input: everything the original controlled interactively is a command-line flag.

## Requirements

* CUDA toolkit 12.x (`nvcc`; Thrust and cuRAND ship with it, nothing else is needed)
* CMake 3.24 or newer
* A C++17 host compiler supported by your CUDA version
* An NVIDIA GPU (compute capability 5.0 or newer) to run; the tests run without one

## Build

```sh
cmake -B build -S .
cmake --build build --config Release
```

CMake compiles for the GPU found in the machine. To target a specific
architecture (e.g. on a build machine without a GPU) pass
`-DCMAKE_CUDA_ARCHITECTURES=86` (RTX 30xx), `89` (RTX 40xx), `80` (A100), ...

## Run

```sh
# one 800x600 frame of the 2p (n=2, l=1, m=0) orbital, 250k particles
./build/atom_realtime

# 120-frame turntable of a 3p (n=3, l=1, m=1) orbital, 100k particles, ray traced
./build/atom_raytracer --frames 120 --spin 3 --out frames_3p

./build/atom_realtime --help      # all flags
```

Frames are written as `frame_00000.ppm`, `frame_00001.ppm`, ... into the output
directory. PPM opens directly in most image viewers and converts with standard
tools:

```sh
ffmpeg -framerate 30 -i frames_3p/frame_%05d.ppm -pix_fmt yuv420p orbital.mp4
magick frames_3p/frame_00000.ppm frame.png          # ImageMagick
python -c "from PIL import Image; Image.open('frames_3p/frame_00000.ppm').save('f.png')"
```

Flags shared by both programs: `--n --l --m --particles --frames --dt --width
--height --radius --azimuth --elevation --spin --seed --out`.
`atom_realtime` adds `--sphere-radius` and `--shaded`; `atom_raytracer` adds
`--sphere-radius` and `--color-scale`. Defaults match the originals.

With `m = 0` the probability current is zero and nothing moves between frames;
use `--spin` to get an animation in that case.

## Tests

`atoms_tests` runs on the CPU (no GPU needed) and checks the code the kernels
are built from: the hydrogen wave functions against textbook values, the
inverse-CDF sampler, the camera transforms, the sphere bounding boxes, and the
BVH build and traversal against brute force.

```sh
./build/atoms_tests
```

## Layout

```
common/
  vec_math.cuh     float3 operators and helpers (host + device)
  hydrogen.cuh     R_nl, P_lm, |psi|^2                 (host + device)
  colormap.cuh     fire heatmap, orbital color         (host + device)
  cdf.cuh          tabulated CDF build + inverse-CDF sampling
  camera.cuh       orbit camera: project / unproject   (host + device)
  geometry.cuh     ray-sphere and ray-box tests        (host + device)
  particles.cuh/cu ParticleCloud: GPU sampling + probability-current kernel
  options.h        command-line flags shared by both programs
  image.h          PPM writer
  cuda_check.cuh   error checking
realtime/
  rasterizer.cuh/cu  depth-tested sphere splatting + grid lines
  main.cu
raytracer/
  bvh.cuh          node layout, BVH view, traversal   (host + device)
  bvh_build.cuh    Morton codes, Karras radix tree, box propagation (host + device)
  bvh.cu           the build kernels and orchestration
  tracer.cuh/cu    primary + shadow rays, shading
  main.cu
tests/
  cpu_tests.cu
```

## How it works

**Sampling** (`particles.cu`). The radial pdf r²R<sub>nl</sub>(r)² and the
polar pdf sin θ P<sub>l</sub><sup>|m|</sup>(cos θ)² are tabulated once on the
CPU into normalized CDFs (4096 and 2048 entries, as in the original). One CUDA
thread per particle draws three uniforms from cuRAND (Philox, seeded per
particle so runs are reproducible), inverts the two CDFs by binary search, picks
φ uniformly, and colors the particle with the fire heatmap.

**Motion.** Each particle stores (r, θ, φ). The probability current of a state
with magnetic quantum number m is a rotation about the y axis with speed
m/(r sin θ); one kernel per frame advances φ accordingly, exactly like the
original's "move along the velocity, re-project onto the sphere" step.

**Realtime renderer** (`rasterizer.cu`). One thread per sphere computes a
conservative pixel bounding box, ray-tests each pixel in it against the sphere,
and depth-tests the hit with a single 64-bit `atomicMin` on a
`(depth << 32) | id` key. A second pass turns the winning ids into colors. Grid
lines are drawn per pixel by distance to the projected segments. Sub-pixel
spheres are drawn as a single dot so they never vanish.

**Ray tracer** (`bvh*.cu*`, `tracer.cu`). The original tests every pixel
against every sphere (O(pixels × spheres) per frame, twice with shadows). This
port builds a linear BVH on the GPU every frame (Karras 2012: Morton codes →
sort → parallel radix tree → bottom-up boxes) and traverses it with an explicit
stack, so a ray costs roughly O(log N) box tests instead of N sphere tests.
Shading is unchanged: ambient + Lambert diffuse from one point light, with a
shadow ray that stops at the first blocker.

## Differences from the original (deliberate)

* **Negative m.** The original evaluates P<sub>l</sub><sup>m</sup> with the
  signed m, which gives the wrong angular shape for m < 0 (the loop building
  P<sub>m</sub><sup>m</sup> only runs for m > 0). This port uses |m|, since
  |Y<sub>l</sub><sup>−m</sup>| = |Y<sub>l</sub><sup>m</sup>|.
* **Sampling precision.** The original returns the left edge of the CDF bin
  (positions snapped to a 4096 × 2048 grid) and integrates the pdf with a left
  Riemann sum. This port interpolates inside the bin and uses the trapezoid
  rule. The means match the analytic ⟨r⟩ to <0.1 %.
* **Particle motion state.** Spherical coordinates are kept instead of
  recovering θ from `acos(y/r)` every frame, so r and θ do not drift.
* **Colormap.** The two originals use slightly different purple stops
  (0.5, 0, 0.99) vs (0.3, 0, 0.6); this port uses the ray tracer's for both.
* **Realtime lighting.** The original's vertex shader computes a Lambert term
  but the fragment shader never applies it, so spheres are flat colored. Flat
  is the default here; `--shaded` applies that term.
* **Tiny spheres.** The realtime original scales a 0.05-unit mesh by n/3,
  giving sub-pixel spheres that OpenGL sometimes drops. The default radius is
  the same, but a sphere that covers no pixel center is drawn as one dot.
* **Fixed quantum numbers per run.** The original rebuilt particles on key
  presses but never rebuilt its cached CDF tables, so the shape after a key
  press was wrong anyway. Here n, l, m are flags and the tables are always
  consistent with them.
* **Seed.** Fixed (`--seed`, default 42) instead of `std::random_device`, so
  renders are reproducible.
