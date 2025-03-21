# cse272-project

<img src="./docs/report/part1.png" alt="Scene 1" width="24%" />
<img src="./docs/report/part2.png" alt="Scene 2" width="24%" />
<img src="./docs/report/part3.png" alt="Scene 3" width="24%" />
<img src="./docs/report/127.0.0.1_8000_.png" alt="Scene 4 (broken)" width="24%" />

WebGPU volumetric renderer for CSE 272.

[Live demo](https://sheeptester.github.io/cse272-project/)
(only tested on Chrome 134, Windows 11).
Click on the canvas to lock your pointer, then use Minecraft controls to move around:

- WASD to move
- Space and shift to fly up/down

<video src="./docs/video.mp4" width="512" height="512" controls></video>

## Development

```shell
$ npm install
$ npm run build
$ npm run dev
```

## References

Homework 2 reference code: https://piazza.com/class/m5ke562n9z969v/post/110
GPU PRNG: https://indico.cern.ch/event/93877/contributions/2118070/attachments/1104200/1575343/acat3_revised_final.pdf
