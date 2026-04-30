import tgpu, { std, d, type TgpuRoot, type TgpuUniform, type TgpuBuffer } from "typegpu";
import { BODY_COUNT_CONST, CelestianBody } from "./data/simulation-data";
import { GRAVITY_MULTIPLIER, ORBIT_PREDICTION_STEPS, ORBIT_PREDICTION_STEPS_CONST, SIMULATION_SPEED } from "./data/settings";
import type { Camera } from "./setup-first-person-camera";
import * as m from "wgpu-matrix";

export function bodiesToArrays(bodies: d.Infer<typeof CelestianBody>[]) {
    const positions = new Float32Array(bodies.length * 3);
    const velocities = new Float32Array(bodies.length * 3);
    bodies.forEach((body, i) => {
        positions[i * 3] = body.position.x;
        positions[i * 3 + 1] = body.position.y;
        positions[i * 3 + 2] = body.position.z;

        velocities[i * 3] = body.velocity.x;
        velocities[i * 3 + 1] = body.velocity.y;
        velocities[i * 3 + 2] = body.velocity.z;
    });

    return { positions, velocities };
}

export function PrepareSimulation(
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
    }[],
    rotationMatricesArray: Float32Array<ArrayBuffer>,
    bodiesRotationMatriciesBuffer: TgpuBuffer<d.WgslArray<d.Mat4x4f>>,
    currentRotationArray: Float32Array<ArrayBuffer>,
) {
    const orbitRenderLayout = tgpu.bindGroupLayout({
        currentBodyIndex: { storage: d.i32, access: "readonly" },
        vertecies: { storage: d.arrayOf(d.vec4f), access: "readonly" },
    });

    const currentBodyIndexBuffer = root.createBuffer(d.i32).$usage("storage", "uniform");
    const orbitVerticiesBuffer = root
        .createBuffer(d.arrayOf(d.vec4f, ORBIT_PREDICTION_STEPS_CONST.$ * BODY_COUNT_CONST.$))
        .$usage("storage");
    const verteciesArray = new Float32Array(ORBIT_PREDICTION_STEPS_CONST.$ * BODY_COUNT_CONST.$ * 4); // CPU BUFFER

    const orbitPrepareRenderBindGroup = root.createBindGroup(orbitRenderLayout, {
        currentBodyIndex: currentBodyIndexBuffer,
        vertecies: orbitVerticiesBuffer,
    });

    const orbitPrepareRenderPipeline = root.createRenderPipeline({
        vertex: tgpu.vertexFn({
            in: { vid: d.builtin.vertexIndex },
            out: { position: d.builtin.position, bodyIndex: d.interpolate("flat", d.i32) },
        })(({ vid }) => {
            "use gpu";
            const bodyIndex = orbitRenderLayout.$.currentBodyIndex;
            const camera = cameraUniform.$;
            const index = (bodyIndex * ORBIT_PREDICTION_STEPS_CONST.$ + vid);
            const point = orbitRenderLayout.$.vertecies[index];

            const position = camera.projection.mul(camera.view).mul(point);

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

    function simulateGravity(positions: Float32Array<ArrayBuffer>, velocities: Float32Array<ArrayBuffer>, bodies: d.Infer<typeof CelestianBody>[], speed: number = SIMULATION_SPEED) {
        for (let i = 0; i < BODY_COUNT_CONST.$; i++) {
            let newVelocity = d.vec3f(0, 0, 0);

            const bodyPosition = d.vec3f(
                positions[i * 3],
                positions[i * 3 + 1],
                positions[i * 3 + 2]
            );

            for (let x = 0; x < BODY_COUNT_CONST.$; x++) {
                if (i === x) continue;
                const otherPosition = d.vec3f(
                    positions[x * 3],
                    positions[x * 3 + 1],
                    positions[x * 3 + 2]
                );
                const otherMass = bodies[x].mass;
                const distance = std.distance(
                    bodyPosition,
                    otherPosition
                );
                const gravityForce = GRAVITY_MULTIPLIER * (otherMass / (distance * distance));
                const direction = std.normalize(otherPosition.sub(bodyPosition));
                newVelocity = newVelocity.add(direction.mul(gravityForce));
            }
            velocities[i * 3] += newVelocity.x * speed;
            velocities[i * 3 + 1] += newVelocity.y * speed;
            velocities[i * 3 + 2] += newVelocity.z * speed;
        }

        for (let i = 0; i < BODY_COUNT_CONST.$; i++) {
            positions[i * 3] += velocities[i * 3] * speed;
            positions[i * 3 + 1] += velocities[i * 3 + 1] * speed;
            positions[i * 3 + 2] += velocities[i * 3 + 2] * speed;
        }
    }

    function predictOrbits(initialPositions: Float32Array<ArrayBuffer>, initialVelocities: Float32Array<ArrayBuffer>, bodies: d.Infer<typeof CelestianBody>[]) {
        const positions = new Float32Array(initialPositions);
        const velocities = new Float32Array(initialVelocities);

        for (let i = 0; i < ORBIT_PREDICTION_STEPS; i++) {
            simulateGravity(positions, velocities, bodies, 10);
            for (let bodyIndex = 0; bodyIndex < BODY_COUNT_CONST.$; bodyIndex++) {
                const vertexIndex = ((bodyIndex * ORBIT_PREDICTION_STEPS) + i) * 4;

                verteciesArray[vertexIndex] = positions[bodyIndex * 3];
                verteciesArray[vertexIndex + 1] = positions[bodyIndex * 3 + 1];
                verteciesArray[vertexIndex + 2] = positions[bodyIndex * 3 + 2];
                verteciesArray[vertexIndex + 3] = 1;
            }
        }

        orbitVerticiesBuffer.write(verteciesArray);
    }

    function renderOrbits(bodies: d.Infer<typeof CelestianBody>[]) {
        bodies.forEach((_, i) => {
            currentBodyIndexBuffer.write(i);

            orbitPrepareRenderPipeline.
                withColorAttachment({ view: context, loadOp: "load", clearValue: { r: 0, g: 0, b: 0, a: 0 } }).
                with(orbitPrepareRenderBindGroup).
                draw(ORBIT_PREDICTION_STEPS, 1);
        });
    }

    function simulateRotation() {
        bodies.forEach((body, i) => {
            const currentAngle = currentRotationArray[i];
            const newAngle = currentAngle + body.rotationSpeed * (Math.PI / 180) * SIMULATION_SPEED;
            currentRotationArray[i] = newAngle;

            const rotationMatrix = m.mat4.rotationY(newAngle);
            rotationMatricesArray.set(rotationMatrix, i * 16);
        });

        bodiesRotationMatriciesBuffer.write(rotationMatricesArray);
    }

    return {
        simulateGravity,
        simulateRotation,
        predictOrbits,
        renderOrbits,
    }
}