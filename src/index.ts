import { captureError } from './utils'
import shaderCode from './shader.wgsl'
import { mat4 } from 'wgpu-matrix'

const canvas = document.getElementById('canvas')
if (!(canvas instanceof HTMLCanvasElement)) {
  throw new TypeError('Failed to find the canvas element.')
}
const context = canvas.getContext('webgpu')!
const format = navigator.gpu.getPreferredCanvasFormat()

const adapter = await navigator.gpu.requestAdapter()
const device = await adapter!.requestDevice({
  requiredFeatures: ['timestamp-query']
})
device.lost.then(info => {
  console.warn('WebGPU device lost. :(', info.message, info)
})
context.configure({ device, format })

const check = captureError(device)

const querySet = device.createQuerySet({
  type: 'timestamp',
  count: 2
})
const resolveBuffer = device.createBuffer({
  size: querySet.count * 8,
  usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
})
const resultBuffer = device.createBuffer({
  size: resolveBuffer.size,
  usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
})

const module = device.createShaderModule({ code: shaderCode })
const { messages } = await module.getCompilationInfo()
if (messages.some(message => message.type === 'error')) {
  console.log(messages)
  throw new SyntaxError('Shader failed to compile.')
}

const pipeline = device.createRenderPipeline({
  layout: 'auto',
  vertex: { module, entryPoint: 'vertex_main' },
  fragment: {
    module,
    entryPoint: 'fragment_main',
    targets: [
      {
        format,
        // https://stackoverflow.com/a/72682494
        blend: {
          color: {
            operation: 'add',
            srcFactor: 'src-alpha',
            dstFactor: 'one-minus-src-alpha'
          },
          alpha: {}
        }
      }
    ]
  }
})

const screenSize = device.createBuffer({
  size: 2 * 4,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
})
device.queue.writeBuffer(
  screenSize,
  0,
  new Float32Array([canvas.width, canvas.height])
)

const sampleToCam = device.createBuffer({
  size: 4 * 4 * 4,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
})
const aspect = canvas.width / canvas.height
const fov = Math.PI / 4
const cot = 1 / Math.tan(fov / 2)
// as tzumao intended
const perspective = mat4.create(
  cot,
  0,
  0,
  0,
  0,
  cot,
  0,
  0,
  0,
  0,
  1,
  -1,
  0,
  0,
  1,
  0
)
const m = mat4.identity()
mat4.scale(m, [-0.5, -0.5 * aspect, 1])
mat4.translate(m, [-1, -1 / aspect, 0])
mat4.multiply(m, perspective)
device.queue.writeBuffer(sampleToCam, 0, mat4.inverse<Float32Array>(m))

const camToWorld = device.createBuffer({
  size: 4 * 4 * 4,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
})
device.queue.writeBuffer(
  camToWorld,
  0,
  mat4.lookAt([0, 0, -3], [0, 0, 0], [0, 1, 0], mat4.create())
)

const uniforms = device.createBindGroup({
  layout: pipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: screenSize } },
    { binding: 1, resource: { buffer: sampleToCam } },
    { binding: 2, resource: { buffer: camToWorld } }
  ]
})

await check()

do {
  const start = performance.now()

  const check = captureError(device)

  const encoder = device.createCommandEncoder()

  const pass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: context.getCurrentTexture().createView(),
        clearValue: [0.2, 0.5, 0.8, 1],
        loadOp: 'clear',
        storeOp: 'store'
      }
    ],
    timestampWrites: {
      querySet,
      beginningOfPassWriteIndex: 0,
      endOfPassWriteIndex: 1
    }
  })
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, uniforms)
  pass.draw(6)
  pass.end()
  device.queue.submit([encoder.finish()])

  resultBuffer.mapAsync(GPUMapMode.READ).then(() => {
    const times = new BigInt64Array(resultBuffer.getMappedRange())
    const gpuTime = times[1] - times[0]
    console.log('gpu time', gpuTime)
    resultBuffer.unmap()
  })

  const cpuTime = performance.now() - start
  console.log('cpu time', cpuTime)

  await Promise.all([check(), new Promise(window.requestAnimationFrame)])
} while (false)
