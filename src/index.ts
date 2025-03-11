import { captureError } from './utils'
import shaderCode from './shader.wgsl'

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
  pass.draw(6)
  pass.end()
  device.queue.submit([encoder.finish()])

  resultBuffer.mapAsync(GPUMapMode.READ).then(() => {
    const times = new BigInt64Array(resultBuffer.getMappedRange())
    const gpuTime = times[1] - times[0]
    resultBuffer.unmap()
  })

  const cpuTime = performance.now() - start

  await Promise.all([check(), new Promise(window.requestAnimationFrame)])
} while (false)
