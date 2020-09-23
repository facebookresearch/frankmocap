# Whole Body Motion Capture Demo (Body + Hand)

Our whole body motion capture is based on our [FrankMocap paper](https://penincillin.github.io/frank_mocap), by intergrating the output of body module and hand module. See our paper for details.

<p>
    <img src="https://github.com/jhugestar/jhugestar.github.io/blob/master/img/frankmocap_wholebody.gif" height="250">
</p>

## Requirements
- You should install both [body module](run_bodymocap.md) and [hand module](run_handmocap.md).


## A Quick Start
- Run the following. The mocap output will be shown on your screen
```
    # Using a machine with a monitor to show output on screen
    # OpenGL renderer is used by default (--renderer_type opengl)
    # The output images are also saved in ./mocap_output
    python -m demo.demo_frankmocap --input_path ./sample_data/single_totalbody.mp4 --out_dir ./mocap_output

    # Screenless mode (e.g., a remote server)
    xvfb-run -a python -m demo.demo_frankmocap --input_path ./sample_data/single_totalbody.mp4 --out_dir ./mocap_output

    # Set other render_type to use other renderers
    python -m demo.demo_frankmocap --input_path ./sample_data/single_totalbody.mp4 --out_dir ./mocap_output --renderer_type pytorch3d
```

## Run Demo with A Webcam Input
- Run,
    ```
        python -m demo.demo_frankmocap --input_path webcam

        #or using opengl gui renderer
        python -m demo.demo_frankmocap --input_path webcam --renderer_type opengl_gui
    ```
- See below to see how to control in opengl gui mode

## Other Details
- Other options should be the same as [body module](run_bodymocap.md). 

## License
- [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode). 
See the [LICENSE](LICENSE) file. 
