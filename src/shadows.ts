import tgpu, { d, std, type StorageFlag, type TgpuBindGroup, type TgpuBindGroupLayout, type TgpuBuffer, type TgpuRoot, type TgpuUniform, type TgpuVertexLayout, type VertexFlag } from "typegpu";
import * as m from "wgpu-matrix";
import { calculateProj, calculateView, Camera } from "./setup-first-person-camera";
import { getVertexAmount } from "./sphere";
import { fullScreenTriangle } from "typegpu/common";
import { DEPTH_BIAS } from "./data/settings";

const FACE_CONFIGS = [
    { name: 'right', dir: d.vec3f(-1, 0, 0), up: d.vec3f(0, 1, 0) },
    { name: 'left', dir: d.vec3f(1, 0, 0), up: d.vec3f(0, 1, 0) },
    { name: 'up', dir: d.vec3f(0, 1, 0), up: d.vec3f(0, 0, -1) },
    { name: 'down', dir: d.vec3f(0, -1, 0), up: d.vec3f(0, 0, 1) },
    { name: 'forward', dir: d.vec3f(0, 0, 1), up: d.vec3f(0, 1, 0) },
    { name: 'backward', dir: d.vec3f(0, 0, -1), up: d.vec3f(0, 1, 0) },
] as const;

export function PrepareShadows(
    root: TgpuRoot,
    canvas: HTMLCanvasElement,
    context: GPUCanvasContext,
    positionVertexLayout: TgpuVertexLayout<d.Disarray<d.float16x4>>,
    bodiesRenderData: {
        verticies: TgpuBuffer<d.WgslArray<d.U32>> & StorageFlag & VertexFlag;
        normals: TgpuBuffer<d.WgslArray<d.U32>> & StorageFlag & VertexFlag;
        trickVerticies: TgpuBuffer<d.Disarray<d.float16x4>> & VertexFlag;
        trickNormals: TgpuBuffer<d.Disarray<d.float16x4>> & VertexFlag;
        debugNormalVerticies: TgpuBuffer<d.WgslArray<d.U32>> & StorageFlag;
        trickDebugNormalVerticies: TgpuBuffer<d.Disarray<d.float16x4>> & VertexFlag;
    }[],
    cameraUniform: TgpuUniform<d.WgslStruct<{
        pos: d.Vec4f;
        targetPos: d.Vec4f;
        view: d.Mat4x4f;
        projection: d.Mat4x4f;
        viewInverse: d.Mat4x4f;
        projectionInverse: d.Mat4x4f;
    }>>,
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
    }>>>,
    mainBindGroupLayout: TgpuBindGroupLayout<{
        positions: {
            storage: (elementCount: number) => d.WgslArray<d.F32>;
            access: "readonly";
        };
        velocities: {
            storage: (elementCount: number) => d.WgslArray<d.F32>;
            access: "readonly";
        };
        rotationMatricies: {
            storage: (elementCount: number) => d.WgslArray<d.Mat4x4f>;
            access: "readonly";
        };
    }>
    ,
    mainBindGroup: TgpuBindGroup<{
        positions: {
            storage: (elementCount: number) => d.WgslArray<d.F32>;
            access: "readonly";
        };
        velocities: {
            storage: (elementCount: number) => d.WgslArray<d.F32>;
            access: "readonly";
        };
        rotationMatricies: {
            storage: (elementCount: number) => d.WgslArray<d.Mat4x4f>;
            access: "readonly";
        };
    }>
    ,
    positions: Float32Array,
    sourceIndex: number

) {
    const shadowMap = root.createTexture({
        size: [1024 * 8, 1024 * 8, 6],
        format: 'depth24plus',
    }).$usage('sampled', "render");

    const depthArrayView = shadowMap.createView(d.textureDepth2dArray(), {
        baseArrayLayer: 0,
        arrayLayerCount: 6,
        aspect: "depth-only"
    });

    const previewSampler = root.createSampler({
        minFilter: 'nearest',
        magFilter: 'nearest',
    });

    const sampler = root.createComparisonSampler({
        magFilter: 'linear',
        minFilter: 'linear',
        mipmapFilter: 'linear',
        compare: "less-equal",
    });

    const shadowsBindGroupLayout = tgpu.bindGroupLayout({
        texture: { texture: d.textureDepthCube() },
        arrayView: { texture: d.textureDepth2dArray() },
        sampler: { sampler: "comparison" },
    });

    const matrixUniform = root.createUniform(d.mat4x4f);
    const faceIndexUniform = root.createUniform(d.i32);
    const currentBodyIndexUniform = root.createUniform(d.i32);
    const sourcePositionUniform = root.createUniform(d.vec3f);
    const depthBiasUniform = root.createUniform(d.f32);
    depthBiasUniform.write(DEPTH_BIAS);

    const shadowsBindGroup = root.createBindGroup(shadowsBindGroupLayout, {
        texture: shadowMap,
        arrayView: depthArrayView,
        sampler,
    });

    const shadowRenderPipeline = root.createRenderPipeline({
        attribs: { inVertex: positionVertexLayout.attrib },
        vertex: tgpu.vertexFn({
            in: { vid: d.builtin.vertexIndex, inVertex: d.vec4f },
            out: {
                position: d.builtin.position,
                worldPosition: d.vec3f,
            },
        })(({ vid, inVertex }) => {
            "use gpu";

            const bodyIndex = currentBodyIndexUniform.$;
            const body = bodiesUniform.$[bodyIndex];

            const faceMatrix = matrixUniform.$;

            const offset = d.vec3f(
                mainBindGroupLayout.$.positions[bodyIndex * 3],
                mainBindGroupLayout.$.positions[bodyIndex * 3 + 1],
                mainBindGroupLayout.$.positions[bodyIndex * 3 + 2],
            );
            const rotationMatrix = mainBindGroupLayout.$.rotationMatricies[bodyIndex];

            const vertex = inVertex.xyz;


            const rotatedPoint = rotationMatrix.mul(d.vec4f(vertex, 1)).xyz;
            const finalPoint = rotatedPoint.mul(body.radius).add(offset);
            const position = faceMatrix.mul(d.vec4f(finalPoint, 1));

            return {
                worldPosition: finalPoint,
                position
            };
        }),
        fragment: tgpu.fragmentFn({ in: { worldPosition: d.vec3f, position: d.builtin.position }, out: d.builtin.fragDepth })(({ worldPosition }) => {
            "use gpu";

            const dist = std.length(worldPosition.sub(sourcePositionUniform.$));
            return (dist + depthBiasUniform.$) / 1000;
        }),
        depthStencil: {
            format: "depth24plus",
            depthWriteEnabled: true,
            depthCompare: "less",
        },
    });

    function renderShadowMaps() {
        const source = d.vec3f(
            positions[sourceIndex * 3],
            positions[sourceIndex * 3 + 1],
            positions[sourceIndex * 3 + 2],
        )
        sourcePositionUniform.write(source);

        FACE_CONFIGS.forEach((face, index) => {

            const view = shadowMap.createView(d.textureDepth2d(), { baseArrayLayer: index, arrayLayerCount: 1 });

            const faceCameraMatrix = m.mat4.mul(
                calculateProj(1, Math.PI / 2, 0.1, 1000),
                calculateView(source, source.add(face.dir), face.up),
            );

            faceIndexUniform.write(index);
            matrixUniform.write(faceCameraMatrix);

            bodiesRenderData.forEach((data, bodyIndex) => {
                if (bodyIndex === sourceIndex) return;

                currentBodyIndexUniform.write(bodyIndex);
                shadowRenderPipeline
                    .withDepthStencilAttachment({
                        view,
                        depthClearValue: 1,
                        depthLoadOp: bodyIndex === 1 ? "clear" : "load",
                        depthStoreOp: "store",
                    })
                    .with(positionVertexLayout, data.trickVerticies)
                    .with(shadowsBindGroup)
                    .with(mainBindGroup)
                    .draw(getVertexAmount())
            });
        });
    }

    const depthToColor = tgpu.fn(
        [d.f32],
        d.vec3f,
    )((depth) => {
        const linear = std.clamp(1 - depth * 6, 0, 1);
        const t = linear * linear;
        const r = std.clamp(t * 2 - 0.5, 0, 1);
        const g = std.clamp(1 - std.abs(t - 0.5) * 2, 0, 0.9) * t;
        const b = std.clamp(1 - t * 1.5, 0, 1) * t;
        return d.vec3f(r, g, b);
    });

    const debugRenderPipeline = root.createRenderPipeline({
        vertex: fullScreenTriangle,
        fragment: ({ uv }) => {
            "use gpu";
            const gridX = d.i32(std.floor(uv.x * 4));
            const gridY = d.i32(std.floor(uv.y * 3));

            const localU = std.fract(uv.x * 4);
            const localV = std.fract(uv.y * 3);
            const localUV = d.vec2f(localU, localV);

            const bgColor = d.vec3f(0.1, 0.1, 0.12);

            let faceIndex = d.i32(-1);

            // Top row: +Y (index 2)
            if (gridY === 0 && gridX === 1) {
                faceIndex = 2;
            }
            // Middle row: -X, +Z, +X, -Z
            if (gridY === 1) {
                if (gridX === 0) {
                    faceIndex = 0; // -X
                }
                if (gridX === 1) {
                    faceIndex = 5; // -Z
                }
                if (gridX === 2) {
                    faceIndex = 1; // +X
                }
                if (gridX === 3) {
                    faceIndex = 4; // +Z
                }
            }
            // Bottom row: -Y (index 3)
            if (gridY === 2 && gridX === 1) {
                faceIndex = 3;
            }

            const depth = std.textureSample(shadowsBindGroupLayout.$.arrayView, previewSampler.$, localUV, faceIndex);

            if (faceIndex < 0) {
                return d.vec4f(bgColor, 1.0);
            }

            const color = depthToColor(depth);

            const border = 0.02;
            const isBorder = localU < border || localU > 1 - border || localV < border || localV > 1 - border;
            const finalColor = std.select(color, std.mul(0.5, color), isBorder);

            return d.vec4f(finalColor, 1.0);
        }
    });

    function debugRender() {
        debugRenderPipeline
            .withColorAttachment({ view: context })
            .with(shadowsBindGroup)
            .draw(3);
    }

    function reloadSettings() {
        depthBiasUniform.write(DEPTH_BIAS);
    }

    return {
        renderShadowMaps,
        debugRender,
        reloadSettings,
        sampler,
        shadowMap,
    };
}