import { d, std, tgpu, type TgpuRoot, type TgpuUniform } from "typegpu";
import { GAUSIAN_ITERATIONS, PIXEL_SCALE } from "../data/settings";

export function PrepareBloom(root: TgpuRoot, canvas: HTMLCanvasElement, context: GPUCanvasContext, pixelScaleUniform: TgpuUniform<d.F32>
) {
    pixelScaleUniform.write(PIXEL_SCALE);

    const blurBindGroundLayout = tgpu.bindGroupLayout({
        isHorizontal: { storage: d.i32, access: "readonly" },
        emmisionTexture: { texture: d.texture2d() },
        sampler: { sampler: "filtering" },
    });

    const sampler = root.createSampler({
        magFilter: "linear",
        minFilter: "linear",
        mipmapFilter: "linear",
    });

    const isBlurHorizontalBuffer = root.createBuffer(d.i32).$usage("storage", "uniform");

    const mainEmmisionTexture = root
        .createTexture({
            size: [canvas.width, canvas.height, 1],
            format: "rgba8unorm",
            mipLevelCount: 4
        })
        .$usage("render", "sampled");

    const currentEmmisionTexture = root
        .createTexture({
            size: [canvas.width, canvas.height, 1],
            format: "rgba8unorm",
            mipLevelCount: 4
        })
        .$usage("render", "sampled");

    const mainBlurBindGroup = root.createBindGroup(blurBindGroundLayout, {
        isHorizontal: isBlurHorizontalBuffer,
        emmisionTexture: mainEmmisionTexture,
        sampler,
    });

    const currentBlurBindGroup = root.createBindGroup(blurBindGroundLayout, {
        isHorizontal: isBlurHorizontalBuffer,
        emmisionTexture: currentEmmisionTexture,
        sampler,
    });

    const blurRenderPipeline = root.createRenderPipeline({
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
            const weights = [0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216];

            let pixelSize = 1.0 / std.textureDimensions(blurBindGroundLayout.$.emmisionTexture).x;
            pixelSize *= pixelScaleUniform.$;

            let result = std
                .textureSampleLevel(
                    blurBindGroundLayout.$.emmisionTexture,
                    blurBindGroundLayout.$.sampler,
                    uv,
                    1
                )

            let sum = d.f32(0);

            result *= std.select(weights[0], 1, blurBindGroundLayout.$.isHorizontal === 0);

            for (let i = -weights.length + 1; i < weights.length; i++) {
                sum += weights[std.abs(i)];

                if (i === 0) continue;

                const vec = std.select(d.vec2f(pixelSize * i, 0), d.vec2f(0, pixelSize * i), blurBindGroundLayout.$.isHorizontal === 0);

                result +=
                    std.textureSampleLevel(
                        blurBindGroundLayout.$.emmisionTexture,
                        blurBindGroundLayout.$.sampler,
                        uv.add(vec),
                        1
                    ).mul(weights[std.abs(i)])
            }

            return result / sum;
        },
        targets: {
            format: "rgba8unorm",
        },
    });

    const finalBloomRenderPipeline = root.createRenderPipeline({
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
                blurBindGroundLayout.$.emmisionTexture,
                blurBindGroundLayout.$.sampler,
                uv,
                1
            );
        },
        targets: {
            blend: {
                color: {
                    srcFactor: "one",
                    dstFactor: "one",
                    operation: "add",
                },
                alpha: {
                    srcFactor: "one",
                    dstFactor: "one",
                    operation: "add",
                }
            },
        }
    });

    function applyGausianBlur() {
        mainEmmisionTexture.generateMipmaps();
        for (let i = 0; i < GAUSIAN_ITERATIONS * 2; i++) {
            const targetView = i % 2 === 0 ? currentEmmisionTexture : mainEmmisionTexture;
            const targetBindGroup = i % 2 === 0 ? mainBlurBindGroup : currentBlurBindGroup;

            isBlurHorizontalBuffer.write(i % 2);
            blurRenderPipeline
                .withColorAttachment({
                    view: targetView.createView("render", { mipLevelCount: 1, baseMipLevel: 1 }),
                    loadOp: "load",
                    clearValue: { r: 0, g: 0, b: 0, a: 1 },
                })
                .with(targetBindGroup)
                .draw(6, 1);
        }
    }

    function render() {
        finalBloomRenderPipeline
            .withColorAttachment({ view: context, loadOp: "load", clearValue: { r: 0, g: 0, b: 0, a: 1 } })
            .with(mainBlurBindGroup)
            .draw(6, 1);
    }

    return {
        emmisionTexture: mainEmmisionTexture,
        applyGausianBlur,
        render
    };
}