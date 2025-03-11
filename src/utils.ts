export function captureError (device: GPUDevice): () => Promise<void> {
  device.pushErrorScope('internal')
  device.pushErrorScope('out-of-memory')
  device.pushErrorScope('validation')

  return async () => {
    const validationError = await device.popErrorScope()
    const memoryError = await device.popErrorScope()
    const internalError = await device.popErrorScope()
    if (validationError) {
      throw new TypeError(
        `WebGPU validation error:\n${validationError.message}`
      )
    }
    if (memoryError) {
      throw new TypeError(`WebGPU out of memory error:\n${memoryError.message}`)
    }
    if (internalError) {
      throw new TypeError(`WebGPU internal error:\n${internalError.message}`)
    }
  }
}
