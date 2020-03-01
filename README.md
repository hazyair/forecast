# Hazyair Forecast

The **_Hazyair_** project consists of several modules:

* Android Application,
* Air Quality Monitoring Tool,
* Dependent components such as:

    - GDPR fragment dialog implementation,
    - Plantower sensor library for node,
    - A Node.js client for dweet.io
    - Wrapper library created to simplify use of SDS011 laser dust sensor,

* Hazyair website,
* Raspberry Pi Zero/Jetson TX1 OpenVINO installer (not published yet).

This repository contains air pollution forecast tool which enables to predict
PM2.5 concentration in the air for the next 4-hours based on 12-hours data.

## Forecast Tool

Forecast Tool is used to define and train prediction model and store it to the
file.

## Conversion Tool

Conversion Tool is used to load the model form the file and convert it to
**_ONNX_** format. **_ONNX_** format can be converted to the **_OpenVINO_**
intermediate representation with following command:

```bash
> mo_onnx.py --input_model forecast.onnx --data_type FP16
```

Conversion Tool is also used to verify if intermediate representation can be
loaded to the **_MYRIAD_** device.

## Known Issues

* https://software.intel.com/en-us/forums/intel-distribution-of-openvino-toolkit/topic/816490,
* https://github.com/opencv/dldt/issues/334

## Disclaimer

OpenVINO is Intel trademark.