import tgpu, { std, d, type TgpuRoot, type TgpuUniform } from "typegpu";
import { BODY_COUNT_CONST, CelestianBody } from "./data/simulation-data";
import { GRAVITY_MULTIPLIER, ORBIT_PREDICTION_STEPS, ORBIT_PREDICTION_STEPS_CONST } from "./data/settings";
import type { Camera } from "./setup-first-person-camera";

export function PrepareSimulation(root: TgpuRoot, canvas: HTMLCanvasElement, context: GPUCanvasContext, cameraUniform: TgpuUniform<typeof Camera>) {
    function simulateGravity(lastBodies: d.Infer<typeof CelestianBody>[]) {
        const finalBodies = lastBodies.map((b) => (CelestianBody(b)));

        finalBodies.forEach((body, i) => {
            let newVelocity = d.vec3f(0, 0, 0);
            finalBodies.forEach((otherBody, x) => {
                if (x === i) return;
                const otherPosition = otherBody.position;
                const otherMass = otherBody.mass;
                const distance = std.distance(body.position, otherPosition);
                const gravityForce = GRAVITY_MULTIPLIER * (otherMass / (distance * distance));
                const direction = std.normalize(otherPosition.sub(body.position));
                newVelocity = newVelocity.add(direction.mul(gravityForce));
            });
            body.velocity = body.velocity.add(newVelocity);
        });

        finalBodies.forEach((body) => {
            body.position = body.position.add(body.velocity);
        });

        return finalBodies;
    }

    const orbitRenderTexture = root
        .createTexture({
            size: [canvas.width, canvas.height, 1],
            format: "rgba8unorm",
        })
        .$usage("render", "sampled");

    const sampler = root.createSampler({
        magFilter: "linear",
        minFilter: "linear",
        mipmapFilter: "linear",
    });

    const orbitPrepareRenderLayout = tgpu.bindGroupLayout({
        currentBodyIndex: { storage: d.i32, access: "readonly" },
        vertecies: { storage: d.arrayOf(d.vec3f), access: "readonly" },
    });

    const orbitFinalRenderLayout = tgpu.bindGroupLayout({
        texture: { texture: d.texture2d(), access: "readonly" },
        sampler: { sampler: "filtering", access: "readonly" },
    });

    const currentBodyIndexBuffer = root.createBuffer(d.i32).$usage("storage", "uniform");

    const orbitVerticiesBuffer = root
        .createBuffer(d.arrayOf(d.vec3f, ORBIT_PREDICTION_STEPS_CONST.$ * BODY_COUNT_CONST.$))
        .$usage("storage", "uniform");

    const orbitPrepareRenderBindGroup = root.createBindGroup(orbitPrepareRenderLayout, {
        currentBodyIndex: currentBodyIndexBuffer,
        vertecies: orbitVerticiesBuffer,
    });

    const orbitFinalRenderBindGroup = root.createBindGroup(orbitFinalRenderLayout, {
        texture: orbitRenderTexture,
        sampler: sampler
    });

    const orbitPrepareRenderPipeline = root.createRenderPipeline({
        vertex: tgpu.vertexFn({
            in: { vid: d.builtin.vertexIndex },
            out: { position: d.builtin.position, bodyIndex: d.interpolate("flat", d.i32) },
        })(({ vid }) => {
            "use gpu";
            const bodyIndex = orbitPrepareRenderLayout.$.currentBodyIndex;
            const camera = cameraUniform.$;
            const point = orbitPrepareRenderLayout.$.vertecies[bodyIndex * ORBIT_PREDICTION_STEPS_CONST.$ + vid];
            const position = camera.projection.mul(camera.view).mul(d.vec4f(point, 1));
            return {
                position,
                bodyIndex,
            };
        }),
        fragment: ({ bodyIndex }) => {
            "use gpu";
            const val = bodyIndex / BODY_COUNT_CONST.$;
            return d.vec4f(val, 1, 1 - val, 1);
        },
        primitive: {
            topology: "line-strip",
        },
        targets: {
            format: "rgba8unorm",
        },
    });

    const finalOrbitRenderPipeline = root.createRenderPipeline({
        vertex: tgpu.vertexFn({
            in: { vid: d.builtin.vertexIndex },
            out: { position: d.builtin.position, uv: d.vec2f },
        })(({ vid }) => {
            "use gpu";
            const positions = [
                d.vec3f(-1, 1, 0),
                d.vec3f(-1, -1, 0),
                d.vec3f(1, -1, 0),
                d.vec3f(-1, 1, 0),
                d.vec3f(1, 1, 0),
                d.vec3f(1, -1, 0),
            ];
            const uvX = positions[vid].x * 0.5 + 0.5;
            const uvY = 1.0 - (positions[vid].y * 0.5 + 0.5);
            return {
                position: d.vec4f(positions[vid], 1),
                uv: d.vec2f(uvX, uvY),
            };
        }),
        fragment: ({ uv }) => {
            "use gpu";
            return std.textureSampleLevel(
                orbitFinalRenderLayout.$.texture,
                orbitFinalRenderLayout.$.sampler,
                uv,
                1
            );
        },
        targets: {
            blend: {
                color: {
                    operation: 'add',
                    srcFactor: 'one-minus-dst-alpha',
                    dstFactor: 'one',
                },
                alpha: {
                    operation: 'add',
                    srcFactor: 'one-minus-dst-alpha',
                    dstFactor: 'one',
                },
            },
        }
    });

    function predictOrbits(bodies: d.Infer<typeof CelestianBody>[]) {
        let currentBodies = bodies.map((b) => (CelestianBody(b)));
        for (let i = 0; i < ORBIT_PREDICTION_STEPS; i++) {
            currentBodies = simulateGravity(currentBodies);

            currentBodies.forEach((body, index) => {
                const vertexIndex = index * ORBIT_PREDICTION_STEPS + i;
                orbitVerticiesBuffer.patch({ [vertexIndex]: body.position });
            });
        }
    }

    function prepareOrbitRender(bodyIndex: number) {
        currentBodyIndexBuffer.write(bodyIndex);
        orbitPrepareRenderPipeline.
            withColorAttachment({ view: orbitRenderTexture, loadOp: bodyIndex === 0 ? "clear" : "load", clearValue: { r: 0, g: 0, b: 0, a: 0 } }).
            with(orbitPrepareRenderBindGroup).
            draw(ORBIT_PREDICTION_STEPS, 1);
    }

    function renderOrbits() {
        finalOrbitRenderPipeline
            .withColorAttachment({ view: context, loadOp: "load", clearValue: { r: 0, g: 0, b: 0, a: 1 } })
            .with(orbitFinalRenderBindGroup)
            .draw(6, 1);
    }

    return {
        simulateGravity,
        predictOrbits,
        prepareOrbitRender,
        renderOrbits
    }
}