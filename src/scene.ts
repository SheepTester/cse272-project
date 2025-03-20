import { Vec3 } from 'wgpu-matrix'

export type Medium = {
  sigmaA: number
  sigmaS: number
}

export type Sphere = {
  center: Vec3
  radius: number

  light?: Light
  interior?: Medium
  exterior?: Medium
}

export type Light = {
  intensity: Vec3
}

export type Scene = {
  media: Medium[]
  shapes: Sphere[]
  cameraMedium: Medium
}

export type SceneData = {
  media: DataView<ArrayBuffer>
  shapes: DataView<ArrayBuffer>
  lights: DataView<ArrayBuffer>
  cameraMedium: Int32Array
}

export function toData ({ media, shapes, cameraMedium }: Scene): SceneData {
  const mediaView = new DataView(new ArrayBuffer(media.length * 4 * 2))
  for (const [i, medium] of media.entries()) {
    mediaView.setFloat32((i * 2 + 0) * 4, medium.sigmaA, true)
    mediaView.setFloat32((i * 2 + 1) * 4, medium.sigmaS, true)
  }

  const lights: { light: Light; shapeIndex: number }[] = []
  const shapesView = new DataView(new ArrayBuffer(shapes.length * 4 * 8))
  for (const [i, sphere] of shapes.entries()) {
    shapesView.setInt32(
      (i * 8 + 0) * 4,
      sphere.light ? lights.length : -1,
      true
    )
    if (sphere.light) {
      lights.push({ light: sphere.light, shapeIndex: i })
    }
    shapesView.setInt32(
      (i * 8 + 1) * 4,
      sphere.interior ? media.indexOf(sphere.interior) : -1,
      true
    )
    shapesView.setInt32(
      (i * 8 + 2) * 4,
      sphere.exterior ? media.indexOf(sphere.exterior) : -1,
      true
    )
    shapesView.setFloat32((i * 8 + 3) * 4, sphere.radius, true)
    shapesView.setFloat32((i * 8 + 4) * 4, sphere.center[0], true)
    shapesView.setFloat32((i * 8 + 5) * 4, sphere.center[1], true)
    shapesView.setFloat32((i * 8 + 6) * 4, sphere.center[2], true)
  }

  const lightsView = new DataView(new ArrayBuffer(lights.length * 4 * 4))
  for (const [i, { light, shapeIndex }] of lights.entries()) {
    lightsView.setFloat32((i * 4 + 0) * 4, light.intensity[0], true)
    lightsView.setFloat32((i * 4 + 1) * 4, light.intensity[1], true)
    lightsView.setFloat32((i * 4 + 2) * 4, light.intensity[2], true)
    lightsView.setInt32((i * 4 + 3) * 4, shapeIndex, true)
  }
  return {
    media: mediaView,
    shapes: shapesView,
    lights: lightsView,
    cameraMedium: new Int32Array([media.indexOf(cameraMedium)])
  }
}
