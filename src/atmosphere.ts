import { common, d, std, tgpu, type RenderFlag, type SampledFlag, type StorageFlag, type TgpuBuffer, type TgpuRoot, type TgpuTexture, type TgpuUniform } from "typegpu";
import type { Camera } from "./setup-first-person-camera";
import { ATMOSPHERE_STEP_COUNT, ATTACHED_BODY_INDEX } from "./data/settings";
import { randf } from "@typegpu/noise";

export function PrepareAtmosphere(
    root: TgpuRoot,
    canvas: HTMLCanvasElement,
    context: GPUCanvasContext,
    cameraUniform: TgpuUniform<typeof Camera>,
    bodies: {
        position: d.v3f;
        radius: number;
        colors: {
            color: d.v4f;
            height: number;
        }[];
        velocity: d.v3f;
        mass: number;
        isSphere: number;
        rotationSpeed: number;
        atmosphere: {
            enabled: number;
            atmosphereRadius: number;
            scatteringStrength: number;
            densityFalloff: number;
            wavelengths: d.v3f;
        };

    }[],
    bodiesUniform: TgpuUniform<d.WgslArray<d.WgslStruct<{
        position: d.Vec3f;
        radius: d.F32;
        colors: d.WgslArray<d.WgslStruct<{
            color: d.Vec4f;
            height: d.F32;
        }>>;
        velocity: d.Vec3f;
        mass: d.F32;
        isSphere: d.U32;
        rotationSpeed: d.F32;
        atmosphere: d.WgslStruct<{
            enabled: d.U32;
            atmosphereRadius: d.F32;
            scatteringStrength: d.F32;
            densityFalloff: d.F32;
            wavelengths: d.Vec3f;
        }>;
    }>>>,
    bodiesPositionsBuffer: TgpuBuffer<d.WgslArray<d.F32>> & StorageFlag,
    depthTexture: TgpuTexture<{
        size: [number, number, 1];
        format: "depth24plus";
    }> & SampledFlag,
    colorTexture: TgpuTexture<{
        size: [number, number, 1];
        format: "rgba8unorm";
    }> & SampledFlag & RenderFlag
) {
    const atmosphereRenderLayout = tgpu.bindGroupLayout({
        bodiesPositionsBuffer: { storage: d.arrayOf(d.f32), access: "readonly" },
        texture: { texture: d.texture2d() },
        sampler: { sampler: "non-filtering" },
        depthTexture: { texture: d.textureDepth2d() },
        colorTexture: { texture: d.texture2d() },
        opticalDepthTexture: { texture: d.texture2d() },
    });

    const currentBodyIndexUniform = root.createUniform(d.u32);
    const stepCountUniform = root.createUniform(d.u32);
    stepCountUniform.write(ATMOSPHERE_STEP_COUNT);

    const sampler = root.createSampler({
        magFilter: "nearest",
        minFilter: "nearest",
    });

    const mainTexture = root
        .createTexture({
            size: [canvas.width, canvas.height, 1],
            format: "rgba8unorm",
        })
        .$usage("render", "sampled");

    const opticalDepthTexture = root.createTexture({
        size: [512, 512, 1],
        format: "rgba8unorm",
    }).$usage("render", "sampled");

    const atmosphereBindGroup = root.createBindGroup(atmosphereRenderLayout, {
        bodiesPositionsBuffer,
        texture: mainTexture,
        sampler,
        depthTexture,
        colorTexture,
        opticalDepthTexture,
    });

    function raySphereIntersect(rayOrigin: d.v3f, rayDirection: d.v3f, sphereCenter: d.v3f, sphereRadius: number) {
        "use gpu";
        const offset = rayOrigin - sphereCenter;

        const a = d.f32(1);
        const b = 2 * std.dot(offset, rayDirection);
        const c = std.dot(offset, offset) - sphereRadius * sphereRadius;
        const discriminant = b * b - 4 * a * c;

        if (discriminant > 0) {
            const s = std.sqrt(discriminant);
            const dstToSphereNear = std.max(0, (-b - s) / (2 * a));
            const dstToSphereFar = (-b + s) / (2 * a);

            if (dstToSphereFar >= 0) {
                return d.vec2f(dstToSphereNear, dstToSphereFar - dstToSphereNear);
            }
        }

        return d.vec2f(12938103293, 0);
    }

    function LinearEyeDepth(depth: number) {
        "use gpu";
        const near = 0.001;
        const far = d.f32(1000);

        return near * far / (far - depth * (far - near));
    }

    function densityAtPoint(point: d.v3f, planetCentre: d.v3f, atmosphereRadius: number, planetRadius: number) {
        "use gpu";
        const densityFalloff = bodiesUniform.$[currentBodyIndexUniform.$].atmosphere.densityFalloff;

        const height = std.length(point - planetCentre) - planetRadius; //height above suface
        const height01 = std.saturate(height / (atmosphereRadius - planetRadius)); // 0 at surface, 1 at top of atmosphere
        const density = std.exp(-height01 * densityFalloff) * (1 - height01);

        return density;
    }

    function opticalDepth(rayOrigin: d.v3f, rayDirection: d.v3f, rayLength: number, planetCentre: d.v3f, atmosphereRadius: number, planetRadius: number) {
        "use gpu";
        let densitySamplePoint = d.vec3f(rayOrigin);
        const stepCount = stepCountUniform.$;
        const stepSize = rayLength / (d.f32(stepCount) - 1);

        let opticalDepth = d.f32(0);

        for (let i = d.u32(0); i < stepCount; i++) {
            const localDensity = densityAtPoint(densitySamplePoint, planetCentre, atmosphereRadius, planetRadius);
            opticalDepth += localDensity * stepSize;
            densitySamplePoint += rayDirection * stepSize;
        }
        return opticalDepth;
    }

    function opticalDepthBaked(rayOrigin: d.v3f, rayDirection: d.v3f, rayLength: number, planetCentre: d.v3f, atmosphereRadius: number, planetRadius: number) {
        "use gpu";
        const height = std.length(rayOrigin - planetCentre) - planetRadius; //height above suface
        const height01 = std.saturate(height / (atmosphereRadius - planetRadius)); // 0 at surface, 1 at top of atmosphere

        const angle = std.dot(rayDirection, std.normalize(rayOrigin - planetCentre));
        const uv = d.vec2f(height01, angle);

        const opticalDepthSample = std.textureSampleLevel(atmosphereRenderLayout.$.opticalDepthTexture, atmosphereRenderLayout.$.sampler, uv, 0);
        return opticalDepthSample.x;
    }

    function calculateLight(rayOrigin: d.v3f, rayDirection: d.v3f, rayLength: number, planetCentre: d.v3f, atmosphereRadius: number, planetRadius: number, originalColor: d.v3f) {
        "use gpu";
        const stepCount = stepCountUniform.$;
        let inScatterPoint = d.vec3f(rayOrigin);
        const stepSize = rayLength / (d.f32(stepCount) - 1);

        let inScatterLight = d.vec3f(0);
        let viewRayOpticalDepth = d.f32(0);

        const sunPosition = d.vec3f(
            atmosphereRenderLayout.$.bodiesPositionsBuffer[0 * 3 + 0],
            atmosphereRenderLayout.$.bodiesPositionsBuffer[0 * 3 + 1],
            atmosphereRenderLayout.$.bodiesPositionsBuffer[0 * 3 + 2],
        );

        const scatteringStrength = bodiesUniform.$[currentBodyIndexUniform.$].atmosphere.scatteringStrength;
        const wavelengths = bodiesUniform.$[currentBodyIndexUniform.$].atmosphere.wavelengths;
        const scatterR = std.pow(600 / wavelengths.x, 4);
        const scatterG = std.pow(600 / wavelengths.y, 4);
        const scatterB = std.pow(600 / wavelengths.z, 4);
        const scatterCoefficients = d.vec3f(scatterR, scatterG, scatterB) * scatteringStrength;

        for (let i = d.u32(0); i < stepCount; i++) {
            const toSunDirection = std.normalize(sunPosition - inScatterPoint);

            const sunRayLength = raySphereIntersect(inScatterPoint, toSunDirection, planetCentre, atmosphereRadius).y;
            const sunRayOpticalDepth = opticalDepth(inScatterPoint, toSunDirection, sunRayLength, planetCentre, atmosphereRadius, planetRadius);
            // const sunRayOpticalDepth = opticalDepthBaked(inScatterPoint, toSunDirection, sunRayLength, planetCentre, atmosphereRadius, planetRadius);
            viewRayOpticalDepth = opticalDepth(inScatterPoint, rayDirection * -1, stepSize * d.f32(i), planetCentre, atmosphereRadius, planetRadius);
            // viewRayOpticalDepth = opticalDepthBaked(inScatterPoint, rayDirection * -1, stepSize * d.f32(i), planetCentre, atmosphereRadius, planetRadius);
            const transmittance = std.exp(-(sunRayOpticalDepth + viewRayOpticalDepth) * scatterCoefficients);
            const localDensity = densityAtPoint(inScatterPoint, planetCentre, atmosphereRadius, planetRadius);
            inScatterLight += localDensity * transmittance * scatterCoefficients * stepSize;
            inScatterPoint += rayDirection * stepSize;
        }

        const originalColTransmittance = std.exp(-viewRayOpticalDepth);

        return originalColor * originalColTransmittance + inScatterLight;
    }

    const atmosphereRenderPipeline = root.createRenderPipeline({
        vertex: common.fullScreenTriangle,
        fragment: ({ uv }) => {
            "use gpu";

            const rayOrigin = cameraUniform.$.pos.xyz;
            const originalColor = d.vec4f(std.textureSample(atmosphereRenderLayout.$.colorTexture, atmosphereRenderLayout.$.sampler, uv).xyz, 1);

            if (bodiesUniform.$[currentBodyIndexUniform.$].atmosphere.enabled === 0) {
                return originalColor;
            }

            const ndc = d.vec4f(
                uv.x * 2.0 - 1.0,
                1.0 - uv.y * 2.0,
                1.0,
                1.0,
            );

            const viewPos = cameraUniform.$.projectionInverse.mul(ndc);
            const viewDir = std.normalize(viewPos.xyz / viewPos.w);

            const viewVector = cameraUniform.$.viewInverse.mul(d.vec4f(viewDir, 0.0)).xyz;
            const rayDirection = std.normalize(viewVector);

            let depth = std.textureSampleLevel(atmosphereRenderLayout.$.depthTexture, atmosphereRenderLayout.$.sampler, uv, 0);
            depth = LinearEyeDepth(depth) * std.length(viewVector);


            const planetCentre = d.vec3f(
                atmosphereRenderLayout.$.bodiesPositionsBuffer[currentBodyIndexUniform.$ * 3 + 0],
                atmosphereRenderLayout.$.bodiesPositionsBuffer[currentBodyIndexUniform.$ * 3 + 1],
                atmosphereRenderLayout.$.bodiesPositionsBuffer[currentBodyIndexUniform.$ * 3 + 2],
            );
            const planetRadius = bodiesUniform.$[currentBodyIndexUniform.$].radius;
            const atmosphereRadius = bodiesUniform.$[currentBodyIndexUniform.$].atmosphere.atmosphereRadius;

            const hitInfo = raySphereIntersect(rayOrigin, rayDirection, planetCentre, atmosphereRadius);

            const dstToAtmosphere = hitInfo.x;
            const dstThrough = std.min(hitInfo.y, depth - dstToAtmosphere);

            if (dstThrough > 0) {
                const pointInAtmosphere = rayOrigin + rayDirection * dstToAtmosphere;
                const light = calculateLight(pointInAtmosphere, rayDirection, dstThrough, planetCentre, atmosphereRadius, planetRadius, originalColor.xyz);
                return d.vec4f(light, 1);
            }

            return originalColor;
        },
    });

    // RAY SPHERE INTERSECTION TEST (looking for mismatches between CPU and GPU implementation)
    function test() {
        const rayOrigin = d.vec3f(0, 2, 0);
        const rayDirection = d.vec3f(0, -1, 0);
        const planetCentre = d.vec3f(0, 0, 0);
        const atmosphereRadius = 3;

        const hitInfo = raySphereIntersect(rayOrigin, rayDirection, planetCentre, atmosphereRadius);
        console.log("cpu", hitInfo.x, hitInfo.y);

        const testPipeline = root.createGuardedComputePipeline(() => {
            "use gpu";
            const rayOrigin = d.vec3f(0, 0, 0);
            const rayDirection = d.vec3f(0, 0, 1);
            const planetCentre = d.vec3f(0, 0, 5);
            const atmosphereRadius = 1.2;

            const hitInfo = raySphereIntersect(rayOrigin, rayDirection, planetCentre, atmosphereRadius);
            console.log("gpu", hitInfo.x, hitInfo.y);
        })

        testPipeline.dispatchThreads();
    }

    function reloadSettings() {
        stepCountUniform.write(ATMOSPHERE_STEP_COUNT);
        // bakeOpticalDepth();
    }

    function render() {
        currentBodyIndexUniform.write(ATTACHED_BODY_INDEX);
        atmosphereRenderPipeline
            .withColorAttachment({ view: context })
            .with(atmosphereBindGroup)
            .draw(3);
    }

    const bakeOpticalDepthPipeline = root.createRenderPipeline({
        vertex: common.fullScreenTriangle,
        fragment: ({ uv }) => {
            "use gpu";
            const planetRadius = bodiesUniform.$[currentBodyIndexUniform.$].radius;
            const atmosphereRadius = bodiesUniform.$[currentBodyIndexUniform.$].atmosphere.atmosphereRadius;

            const height01 = uv.x; // from 0 (surface) to 1 (top of atmosphere)
            const angle = uv.y * d.f32(Math.PI); // from 1 (looking up) to 0 (looking down)
            const dir = d.vec2f(std.sin(angle), std.cos(angle));

            const inPoint = d.vec2f(0, std.mix(planetRadius, atmosphereRadius, height01));
            const dstThrough = raySphereIntersect(d.vec3f(inPoint, 0), d.vec3f(dir, 0), d.vec3f(0), atmosphereRadius).y;
            const outScattering = opticalDepth(d.vec3f(inPoint, 0), d.vec3f(dir, 0), dstThrough, d.vec3f(0), atmosphereRadius, planetRadius);
            return d.vec4f(d.vec3f(outScattering), 1);
        },
        targets: { format: "rgba8unorm" },
    });

    function testRender() {
        bakeOpticalDepth();

        const tLayout = tgpu.bindGroupLayout({
            texture: { texture: d.texture2d() },
            sampler: { sampler: "non-filtering" },
        });

        const testRenderPipeline = root.createRenderPipeline({
            vertex: common.fullScreenTriangle,
            fragment: ({ uv }) => {
                "use gpu";

                const color = std.textureSample(tLayout.$.texture, tLayout.$.sampler, uv);
                return color;
            }
        });

        const tBindGroup = root.createBindGroup(tLayout, {
            texture: opticalDepthTexture,
            sampler: sampler,
        });

        testRenderPipeline
            .withColorAttachment({ view: context })
            .with(tBindGroup)
            .draw(3);
    }

    function bakeOpticalDepth() {
        currentBodyIndexUniform.write(3); // TODO: loop through bodies

        bakeOpticalDepthPipeline
            .withColorAttachment({ view: opticalDepthTexture })
            .draw(3);
    }

    // bakeOpticalDepth();

    return {
        test,
        render,
        reloadSettings,
        testRender
    }
}