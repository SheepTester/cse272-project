import { captureError } from './utils'
import shaderCode from './shader.wgsl'
import { mat4, vec3 } from 'wgpu-matrix'
import { Medium, Scene, toData } from './scene'
import { scene } from './scenes/volpath-test3'

if (!navigator.gpu) {
  alert('Your browser doesnt support WebGPU, or it is not enabled.')
}

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
mat4.transpose(perspective, perspective)
const m = mat4.identity()
mat4.scale(m, [-0.5, -0.5 * aspect, 1], m)
mat4.translate(m, [-1, -1 / aspect, 0], m)
mat4.multiply(m, perspective, m)
device.queue.writeBuffer(sampleToCam, 0, mat4.inverse<Float32Array>(m))

const camToWorld = device.createBuffer({
  size: 4 * 4 * 4,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
})
device.queue.writeBuffer(
  camToWorld,
  0,
  mat4.aim([0, 0, -4], [0, 0, 0], [0, 1, 0], mat4.create())
)

const { media, shapes, lights, cameraMedium } = toData(scene)

const mediaBuffer = device.createBuffer({
  size: media.buffer.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
})
const shapesBuffer = device.createBuffer({
  size: shapes.buffer.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
})
const lightsBuffer = device.createBuffer({
  size: lights.buffer.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
})
const cameraMediumBuffer = device.createBuffer({
  size: cameraMedium.buffer.byteLength,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
})
device.queue.writeBuffer(mediaBuffer, 0, media.buffer)
device.queue.writeBuffer(shapesBuffer, 0, shapes.buffer)
device.queue.writeBuffer(lightsBuffer, 0, lights.buffer)
device.queue.writeBuffer(cameraMediumBuffer, 0, cameraMedium.buffer)

const uniforms = device.createBindGroup({
  layout: pipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: screenSize } },
    { binding: 1, resource: { buffer: sampleToCam } },
    { binding: 2, resource: { buffer: camToWorld } },
    { binding: 3, resource: { buffer: mediaBuffer } },
    { binding: 4, resource: { buffer: shapesBuffer } },
    { binding: 5, resource: { buffer: lightsBuffer } },
    { binding: 6, resource: { buffer: cameraMediumBuffer } }
  ]
})

await check()

const camera = {
  x: 0,
  y: 0,
  z: -3,
  xv: 0,
  yv: 0,
  zv: 0,
  rx: 0,
  ry: 0
}
const keys = {
  forward: false,
  backward: false,
  left: false,
  right: false,
  up: false,
  down: false
}
const keyMap: Record<string, keyof typeof keys> = {
  w: 'forward',
  a: 'left',
  s: 'backward',
  d: 'right',
  ' ': 'up',
  shift: 'down'
}
document.addEventListener('keydown', e => {
  const key = keyMap[e.key.toLowerCase()]
  if (key) {
    keys[key] = true
  }
})
document.addEventListener('keyup', e => {
  const key = keyMap[e.key.toLowerCase()]
  if (key) {
    keys[key] = false
  }
})
document.addEventListener('blur', () => {
  for (const key of Object.values(keyMap)) {
    keys[key] = false
  }
})
canvas.addEventListener('click', async () => {
  await canvas.requestPointerLock({ unadjustedMovement: true })
})
canvas.addEventListener('mousemove', e => {
  if (document.pointerLockElement === canvas) {
    handleMouseMove(e)
  }
})
type DragState = { pointerId: number; lastX: number; lastY: number }
let dragState: DragState | null = null
document.addEventListener('pointerdown', e => {
  if (e.pointerType === 'touch' && !dragState) {
    dragState = { pointerId: e.pointerId, lastX: e.clientX, lastY: e.clientY }
    canvas.setPointerCapture(e.pointerId)
  }
})
canvas.addEventListener('pointermove', e => {
  if (e.pointerId === dragState?.pointerId) {
    const movementX = e.clientX - dragState.lastX
    const movementY = e.clientY - dragState.lastY
    handleMouseMove({ movementX, movementY })
    dragState.lastX = e.clientX
    dragState.lastY = e.clientY
  }
})
const handlePointerEnd = (e: PointerEvent) => {
  if (e.pointerId === dragState?.pointerId) {
    dragState = null
  }
}
canvas.addEventListener('pointerup', handlePointerEnd)
canvas.addEventListener('pointercancel', handlePointerEnd)
function handleMouseMove ({
  movementX,
  movementY
}: {
  movementX: number
  movementY: number
}) {
  camera.rx += movementY / 400
  camera.ry -= movementX / 400
}

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

  const bruh = resultBuffer.mapAsync(GPUMapMode.READ).then(() => {
    const times = new BigInt64Array(resultBuffer.getMappedRange())
    const gpuTime = times[1] - times[0]
    // console.log('gpu time', gpuTime)
    resultBuffer.unmap()
  })

  const cpuTime = performance.now() - start
  // console.log('cpu time', cpuTime)

  await Promise.all([check(), new Promise(window.requestAnimationFrame), bruh])

  // scene.shapes[0].center = vec3.fromValues(
  //   0,
  //   Math.sin(Date.now() / 1000),
  //   Math.cos(Date.now() / 1789) + 1
  // )
  // scene.shapes[1].center = vec3.fromValues(
  //   Math.sin(Date.now() / -2837) * 5,
  //   0,
  //   Math.cos(Date.now() / -2837) * 5
  // )

  const { media, shapes, lights, cameraMedium } = toData(scene)
  device.queue.writeBuffer(mediaBuffer, 0, media.buffer)
  device.queue.writeBuffer(shapesBuffer, 0, shapes.buffer)
  device.queue.writeBuffer(lightsBuffer, 0, lights.buffer)
  device.queue.writeBuffer(cameraMediumBuffer, 0, cameraMedium.buffer)

  /** 1/s */
  const DRAG_CONST = 2
  /** world units / s */
  const MOVE_CONST = 0.3
  const t = Math.min((performance.now() - start) / 1000, 0.1)
  const newVelocity = { x: 0, y: 0, z: 0 }
  // vel += -drag * vel * time
  // vel = vel - drag * time * vel = (1 - drag * time) * vel
  newVelocity.x = Math.max(1 - DRAG_CONST * t, 0) * camera.xv
  newVelocity.y = Math.max(1 - DRAG_CONST * t, 0) * camera.yv
  newVelocity.z = Math.max(1 - DRAG_CONST * t, 0) * camera.zv
  let moveX = 0
  let moveZ = 0
  if (keys.left) moveX += MOVE_CONST
  if (keys.right) moveX += -MOVE_CONST
  if (keys.down) newVelocity.y += -MOVE_CONST
  if (keys.up) newVelocity.y += MOVE_CONST
  if (keys.backward) moveZ += -MOVE_CONST
  if (keys.forward) moveZ += MOVE_CONST
  newVelocity.x += moveX * Math.cos(-camera.ry) - moveZ * Math.sin(-camera.ry)
  newVelocity.z += moveX * Math.sin(-camera.ry) + moveZ * Math.cos(-camera.ry)
  camera.x += t * ((camera.xv + newVelocity.x) / 2)
  camera.y += t * ((camera.yv + newVelocity.y) / 2)
  camera.z += t * ((camera.zv + newVelocity.z) / 2)
  camera.xv = newVelocity.x
  camera.yv = newVelocity.y
  camera.zv = newVelocity.z

  const cameraTransform = mat4.identity()
  mat4.translate(
    cameraTransform,
    [camera.x, camera.y, camera.z],
    cameraTransform
  )
  mat4.rotateY(cameraTransform, camera.ry, cameraTransform)
  mat4.rotateX(cameraTransform, camera.rx, cameraTransform)
  device.queue.writeBuffer(camToWorld, 0, cameraTransform)
} while (false)
