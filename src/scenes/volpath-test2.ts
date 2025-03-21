import { vec3 } from 'wgpu-matrix'
import { Medium, Scene } from '../scene'

const medium: Medium = { sigmaA: 0.1, sigmaS: 0.7 }
export const scene: Scene = {
  shapes: [
    {
      center: vec3.fromValues(0, 0, 0),
      radius: 1,
      hasMaterial: true,
      exterior: medium,
      light: {
        intensity: vec3.fromValues(0.4, 2.32, 3.2)
      }
    },
    {
      center: vec3.fromValues(-3, 0, -1.5),
      radius: 1.5,
      hasMaterial: true,
      exterior: medium,
      light: {
        intensity: vec3.fromValues(24, 10, 24)
      }
    }
  ],
  cameraMedium: medium,
  maxDepth: 2
}
