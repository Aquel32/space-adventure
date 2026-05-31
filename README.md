# space-adventure

3D Planetary system simulation build with [TypeGPU](https://github.com/software-mansion/TypeGPU) 

During this project I have set my first steps into Graphic Programming world, learning some basic rendering concepts and visual effects.

## Demo

You can check it out yourself [here](https://aquel32.github.io/space-adventure/).

<div style="display: flex; justify-content: space-between">
  <img width="24%" alt="b" src="https://github.com/user-attachments/assets/178985ee-f8a4-4996-9523-dd38d766a618" />
  <img width="24%" alt="a" src="https://github.com/user-attachments/assets/8eb71218-43a5-46cd-8713-b982a1ff230e" />
  <img width="24%" alt="d" src="https://github.com/user-attachments/assets/3d63d7b3-60f4-404b-9c7d-c4fad76ece47" />
  <img width="24%" alt="c" src="https://github.com/user-attachments/assets/74ea7d6f-1c6a-4455-b96c-f215c7ade507" />
</div>

https://github.com/user-attachments/assets/a0a99e05-1a62-470f-ad8d-d28446be95ec

## How it works

### Geometry and Phong Lighting

Celestial Bodies are generated with **cubesphere** technique with a **Perlin Noise** offset applied to create uneven terrain.
By varying surface colors, we get diffrent-looking planets.

Normal vectors are analytically calculated by sampling Perlin Noise around vertex. Then they are using to compute **Phong lighting**.

### Bloom

Pixels that cross a brightness treshhold are simultaneously drawn onto emission texture.<br>
That texture is blurred using 2 pass **Gaussian Blur**.
To limit cost of many iterations, I use **Mipmapping** to apply blur on lower-resolution versions of texture.<br>
To avoid unnecessary memory allocation, pipeline **ping-pongs** between two textures.

### Shadows

Celestian bodies need to cast shodows on one another.<br>
Shadows are generated using **Omnidirectional Shadow Mapping**.<br>
A point light (the Sun) hosts six cameras pointing in all direction of a cube's faces, rendering the scene's depth into a **depth cube**.
In main render pass, we sample that depth cube to determine if a pixel is in shadow (it's just multiplying final Phong lighting).

> Currently, body index `0` is hardcoded as the only light emmiter.
> Additionally, default planet scales are quite small relative to the distances between them, making planetary shadows not that visible.<br>
> If you want to see the effect, it's better to increase up planet radiuses.

### Atmosphere

The atmosphere effect uses **raymarching**.<br>
At each sample point along camera ray, shader checks average atmospheric density from that point toward the sun.
Summing and normalizing theese values results in beautiful light multiplier.<br>
My implementation is heavily based on [Sebastian Lague's](https://www.youtube.com/watch?v=DxfEbulyFcY) video, rewritten to TypeGPU.

> Right now, planet density pre-computation isn't implemented.<br>
> Also the atmosphere is currently simulated only for the closest body to the camera, which looks weird when you look at Body 3 while standing on Body 4.

### Other optimization

Verticies and normal positions are stored in **Half-Precision Float**. <br>
Theese compact vectors are fed directly into the main and shadow render pipelines via **Vertex Layouts**. <br> 
That way we cut vertex memory allocation by half.

### Quick note on gravity

Gravity wasn't the primary focus of this project, so it's a bit clunky and can be unstable.<br>
All simulation takes place on CPU using a simple gravitational acceleration formula.<br>
Orbit prediction works by running the simulation hundreds of times on a cloned dataset (curently hardcoded to `10,000` steps ahead) and saving coordinates into a buffer.<br>
Orbit render pipeline uses `line-strip` topology, meaning each vertex connects to the previous one to draw a path.

## Known Issues

- TODO: Render multiple atmospheres simultaneously
- TODO: Implement a pre-computed density lookup texture for atmosphere performance.
- TODO: Configure atmosphere settings for the rest of planets.
- ISSUE: CPU Perlin Noise doesn't use same alogrithm as GPU (`XOROSHIRO64**`).
- ISSUE: "Sliding" behaviour when standing on planet while `Pull Camera` option is enabled.
