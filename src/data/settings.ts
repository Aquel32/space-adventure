import tgpu, { d } from "typegpu";
import { INITIAL_BODIES } from "./simulation-data";
import { moveCameraToAttachedObject, pixelScaleUniform } from "../main";

export let GRAVITY_MULTIPLIER = 0.04;
export function SetGravityMultiplier(newGravityMultiplier: number) {
    GRAVITY_MULTIPLIER = newGravityMultiplier;
}

export let SIMULATION_SPEED = 0;
export function SetSimulationSpeed(newSimulationSpeed: number) {
    SIMULATION_SPEED = newSimulationSpeed;
}

export let GAUSIAN_ITERATIONS = d.f32(5);
export function SetGausianIterations(newGaussianIterations: number) {
    GAUSIAN_ITERATIONS = newGaussianIterations;
}

export let PIXEL_SCALE = d.f32(1);
export function SetPixelScale(newPixelScale: number) {
    PIXEL_SCALE = newPixelScale;
    pixelScaleUniform.write(PIXEL_SCALE);
}

export let RENDER_ORBITS = true;
export function SetRenderOrbits(newRenderOrbits: boolean) {
    RENDER_ORBITS = newRenderOrbits;
}

export let DEBUG_NORMALS = false;
export function SetDebugNormals(newDebugNormals: boolean) {
    DEBUG_NORMALS = newDebugNormals;
}

export let SHOW_DEPTH_CUBE = false;
export function SetShowDepthCube(newShowDepthCube: boolean) {
    SHOW_DEPTH_CUBE = newShowDepthCube;
}

export let DEBUG_SHADOWS = false;
export function SetDebugShadows(newDebugShadows: boolean) {
    DEBUG_SHADOWS = newDebugShadows;
}

export const ORBIT_PREDICTION_STEPS = 100;
export const ORBIT_PREDICTION_STEPS_CONST = tgpu.const(d.i32, ORBIT_PREDICTION_STEPS);

export let ATTACHED_BODY_INDEX = -1;
export function SetAttachedBody(newIndex: number) {
    if (newIndex == -1) {
        ATTACHED_BODY_INDEX = -1;
        return;
    }

    if (newIndex < 0 || newIndex >= INITIAL_BODIES.length) return;

    ATTACHED_BODY_INDEX = newIndex;
    moveCameraToAttachedObject();
}

export const SPHERE_DIVISIONS = 5;

export let DEPTH_BIAS = 0.1;
export function SetDepthBias(newDepthBias: number) {
    DEPTH_BIAS = newDepthBias;
}

export let NORMAL_OFFSET = 0.6;
export function SetNormalOffset(newNormalOffset: number) {
    NORMAL_OFFSET = newNormalOffset;
}