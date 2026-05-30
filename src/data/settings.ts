import tgpu, { d } from "typegpu";

export function prepareSettings()
{
     let GRAVITY_MULTIPLIER = 0.04;
     function SetGravityMultiplier(newGravityMultiplier: number) {
        GRAVITY_MULTIPLIER = newGravityMultiplier;
    }

     let SIMULATION_SPEED = 1;
     function SetSimulationSpeed(newSimulationSpeed: number) {
        SIMULATION_SPEED = newSimulationSpeed;
    }

     let GAUSIAN_ITERATIONS = d.f32(5);
     function SetGausianIterations(newGaussianIterations: number) {
        GAUSIAN_ITERATIONS = newGaussianIterations;
    }

     let PIXEL_SCALE = d.f32(1);
     function SetPixelScale(newPixelScale: number) {
        PIXEL_SCALE = newPixelScale;
    }

     let RENDER_ORBITS = true;
     function SetRenderOrbits(newRenderOrbits: boolean) {
        RENDER_ORBITS = newRenderOrbits;
    }

     let DEBUG_NORMALS = false;
     function SetDebugNormals(newDebugNormals: boolean) {
        DEBUG_NORMALS = newDebugNormals;
    }

     let SHOW_DEPTH_CUBE = false;
     function SetShowDepthCube(newShowDepthCube: boolean) {
        SHOW_DEPTH_CUBE = newShowDepthCube;
    }

     let DEBUG_SHADOWS = false;
     function SetDebugShadows(newDebugShadows: boolean) {
        DEBUG_SHADOWS = newDebugShadows;
    }

     const ORBIT_PREDICTION_STEPS = 1000;
     const ORBIT_PREDICTION_STEPS_CONST = tgpu.const(d.i32, ORBIT_PREDICTION_STEPS);

    let moveCameraToAttachedObjectCallback: ()=>void;
    function SetAttachedBodyCallback(moveCameraFunction: ()=>void)
    {
        moveCameraToAttachedObjectCallback = moveCameraFunction;
    }

     let ATTACHED_BODY_INDEX = -1;
     function SetAttachedBody(newIndex: number, maxIndex: number, moveCamera: boolean = true) {

        if (newIndex < 0 || newIndex >= maxIndex || newIndex === -1) {
            ATTACHED_BODY_INDEX = -1;
            document.querySelector(".ab")!.setAttribute("value", "-1");
            return;
        }

        ATTACHED_BODY_INDEX = newIndex;

        if (moveCamera && moveCameraToAttachedObjectCallback) {
            moveCameraToAttachedObjectCallback();
        }

        document.querySelector(".ab")!.setAttribute("value", `${newIndex}`);
    }

     const SPHERE_DIVISIONS = 4;

     let DEPTH_BIAS = 0.1;
     function SetDepthBias(newDepthBias: number) {
        DEPTH_BIAS = newDepthBias;
    }

     let NORMAL_OFFSET = 0.6;
     function SetNormalOffset(newNormalOffset: number) {
        NORMAL_OFFSET = newNormalOffset;
    }

     let ATMOSPHERE_STEP_COUNT = 10;
     function SetAtmosphereStepCount(newAtmosphereStepCount: number) {
        ATMOSPHERE_STEP_COUNT = newAtmosphereStepCount;
    }

     let ATMOSPHERE_SHOW_PREBAKED_DEPTH = false;
     function SetShowPrebakedDepth(newShowPrebakedDepth: boolean) {
        ATMOSPHERE_SHOW_PREBAKED_DEPTH = newShowPrebakedDepth;
    }

     let PERLIN_STRENGTH = 0.3;
     let PERLIN_EPSILON = 0.01;

     function SetStrength(newStrength: number) {
    PERLIN_STRENGTH = newStrength;
    }

     function SetEpsilon(newEpsilon: number) {
    PERLIN_EPSILON = newEpsilon;
    }

     let PULL_CAMERA = false;
     function SetPullCamera(newPullCamera: boolean) {
        PULL_CAMERA = newPullCamera;
    }

    return {
        get GRAVITY_MULTIPLIER() { return GRAVITY_MULTIPLIER; },
        get SIMULATION_SPEED() { return SIMULATION_SPEED; },
        get GAUSIAN_ITERATIONS() { return GAUSIAN_ITERATIONS; },
        get PIXEL_SCALE() { return PIXEL_SCALE; },
        get RENDER_ORBITS() { return RENDER_ORBITS; },
        get DEBUG_NORMALS() { return DEBUG_NORMALS; },
        get SHOW_DEPTH_CUBE() { return SHOW_DEPTH_CUBE; },
        get DEBUG_SHADOWS() { return DEBUG_SHADOWS; },
        ORBIT_PREDICTION_STEPS,
        ORBIT_PREDICTION_STEPS_CONST,
        get ATTACHED_BODY_INDEX() { return ATTACHED_BODY_INDEX; },
        get DEPTH_BIAS() { return DEPTH_BIAS; },
        get NORMAL_OFFSET() { return NORMAL_OFFSET; },
        get ATMOSPHERE_STEP_COUNT() { return ATMOSPHERE_STEP_COUNT; },
        get ATMOSPHERE_SHOW_PREBAKED_DEPTH() { return ATMOSPHERE_SHOW_PREBAKED_DEPTH; },
        get PERLIN_STRENGTH() { return PERLIN_STRENGTH; },
        get PERLIN_EPSILON() { return PERLIN_EPSILON; },
        get PULL_CAMERA() { return PULL_CAMERA; },
        SetGravityMultiplier,
        SetSimulationSpeed,
        SetGausianIterations,
        SetPixelScale,
        SetRenderOrbits,
        SetDebugNormals,
        SetShowDepthCube,
        SetDebugShadows,
        SetAttachedBody,
        SPHERE_DIVISIONS,
        SetDepthBias,
        SetNormalOffset,
        SetAtmosphereStepCount,
        SetShowPrebakedDepth,
        SetStrength,
        SetEpsilon,
        SetPullCamera,
        SetAttachedBodyCallback,
    };
}