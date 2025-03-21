import { vec3 } from 'wgpu-matrix'
import { Medium, Scene } from '../scene'

const medium1: Medium = { sigmaA: 1 * 0.05, sigmaS: 5 * 0.05 }
const medium2: Medium = { sigmaA: 3, sigmaS: 0.5 }
export const scene: Scene = {
  media: [medium1],
  shapes: [
    {
      center: vec3.fromValues(-0.5, -0.5, 0),
      radius: 0.75,
      interior: medium2,
      exterior: medium1
    },
    {
      center: vec3.fromValues(1, 1, 2),
      radius: 2,
      exterior: medium1,
      light: {
        intensity: vec3.fromValues(0.4, 2.32, 3.2)
      }
    }
  ],
  cameraMedium: medium1
}
