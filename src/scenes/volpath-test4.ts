import { vec3 } from 'wgpu-matrix'
import { Medium, Scene } from '../scene'

const medium1: Medium = { sigmaA: 1 * 0.05, sigmaS: 5 * 0.05 }
const medium2: Medium = { sigmaA: 1, sigmaS: 0.5 }
export const scene: Scene = {
  shapes: [
    {
      center: vec3.fromValues(0, 0, 0),
      radius: 0.75,
      hasMaterial: false,
      interior: medium2,
      exterior: medium1
    },
    {
      center: vec3.fromValues(1.5, 1.5, 2),
      radius: 0.25,
      hasMaterial: true,
      exterior: medium1,
      light: {
        intensity: vec3.fromValues(8, 46.4, 64)
      }
    },
    {
      center: vec3.fromValues(-1.5, -1.5, 2),
      radius: 2,
      hasMaterial: true,
      exterior: medium1,
      light: {
        intensity: vec3.fromValues(2.4, 1, 2.4)
      }
    }
  ],
  cameraMedium: medium1,
  maxDepth: 6
}
