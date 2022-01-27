//! Tensorflow JS library
import * as tf from '@tensorflow/tfjs';
import { PsySegBuf, ColorSpaceType } from '../PsySeg.js';
import { preProcessing } from "../Preprocessing/Preprocessing.js";
import { simd } from  "wasm-feature-detect";
import createTFLiteModule from "../model/tflite-import/tflite.mjs"
import createTFLiteSIMDModule from "../model/tflite-import/tflite-simd.mjs"
//! Strict mode
"use strict";

//! Model variables
var TfLiteModel = null;
var TfJsModel = null;

//! ML class for storing segmentation techniques
export class SmartEyes {

  //! Constructor for SmartEyes
  constructor(backend, reload) {

    //! Init Time
    this.startTime = Date.now();
    this.initTime = 0;

    //! ML model
    this.model = null;

    //! Type of model
    this.type = backend === 'tflite' ? "tflite" : "tfjs";

    //! Inference Size
    this.segWidth = backend === 'tflite' ? 160 : 192;
    this.segHeight = backend === 'tflite' ? 96 : 192;

    //! Check reload model
    if (!reload) {

      if ((this.type === "tflite") && (TfLiteModel !== null)) {

        // Re-assign DNN model
        this.model = TfLiteModel;

        // Loading model
        this.model._loadModel(407232);

        // Bind methods
        this.runTFJSInference = this.runTFJSInference.bind(this);
        this.runTFLiteInference = this.runTFLiteInference.bind(this);
        this.loadTfLiteModel = this.loadTfLiteModel.bind(this);
        this.loadTfJsModel = this.loadTfJsModel.bind(this);
        this.run = this.run.bind(this);

        return;

      } else if ((this.type === "tfjs") && (TfJsModel !== null)) {

        // Re-assign DNN model
        this.model = TfJsModel;

        // Bind methods
        this.runTFJSInference = this.runTFJSInference.bind(this);
        this.runTFLiteInference = this.runTFLiteInference.bind(this);
        this.loadTfLiteModel = this.loadTfLiteModel.bind(this);
        this.loadTfJsModel = this.loadTfJsModel.bind(this);
        this.run = this.run.bind(this);

        return;

      }

    }

    //! Check backend
    if (backend === "tflite") {

      //! Firstly, check SIMD support from browser
      simd()
        .then((simdSupported) => {
          if (simdSupported) {

            //! Case 1: SIMD is supported
            console.log("SIMD is supported by browser");

            //! Loading model
            createTFLiteSIMDModule()
              .then((module) => {
                this.loadTfLiteModel(module);
              })
              .catch((e) => {

                //! Init Time
                this.initTime = Date.now() - this.startTime;

                //! Logging failed
                console.log("Create tflite module failed due to " + e);
              });

          } else {

            //! Case 2: SIMD is not supported
            console.log("SIMD is not supported by browser");

            //! Loading model
            createTFLiteModule()
              .then((module) => {
                this.loadTfLiteModel(module);
              })
              .catch((e) => {

                //! Init Time
                this.initTime = Date.now() - this.startTime;

                //! Logging failed
                console.log("Create tflite module failed due to " + e);
              });
          }
        })
        .catch((e) => {

          //! Case 3: Cannot check SIMD support
          console.log("Cannot check SIMD support by browsers due to " + e);

          //! Loading model
          createTFLiteModule()
            .then((module) => {
              this.loadTfLiteModel(module);
            })
            .catch((e) => {

              //! Init Time
              this.initTime = Date.now() - this.startTime;

              //! Logging failed
              console.log("Create tflite module failed due to " + e);
            });
        })

    } else {

      this.loadTfJsModel();

    }

    // Bind methods
    this.runTFJSInference = this.runTFJSInference.bind(this);
    this.runTFLiteInference = this.runTFLiteInference.bind(this);
    this.loadTfLiteModel = this.loadTfLiteModel.bind(this);
    this.loadTfJsModel = this.loadTfJsModel.bind(this);
    this.run = this.run.bind(this);
  }

  // Loading TFJS model
  loadTfJsModel() {

    //! Change type of model & size
    //let modelLink = "http://localhost:8080/model/tfjs/model.json";
    let modelLink = "https://psyjs-cdn.nuvixa.com/model.json";

    //! Assign headers for model loading
    let myHeaders = new Headers();
    let myInit = {
      method: 'GET',
      headers: myHeaders,
      mode: 'cors',
      cache: 'default'
    };

    //! Loading model
    tf.loadGraphModel(modelLink, { requestInit: myInit })
      .then((model) => {

        //! Set backend
        /*
        switch (backend) {
            case 'gpu': tf.setBackend('webgl');
                       break;
            case 'cpu': tf.setBackend('cpu');
                        break;
            case 'wasm': tf.setBackend('wasm');
                         break;
            default: break;
        }
        */

        //! Set backend --> always uses "webgl" (GPU)
        tf.setBackend('webgl');

        //! Get model
        this.model = model;
        TfJsModel = model;

        //! Log backend
        console.log("Current backend is: " + tf.getBackend());

        //! Init Time
        this.initTime = Date.now() - this.startTime;
      })
      .catch((e) => {

        //! Init Time
        this.initTime = Date.now() - this.startTime;

        //! Logging failed
        console.log("Loading model error due to " + e);
      });
  }

  // Loading TFlite model
  loadTfLiteModel(module) {

    // let module 
    //! Get tflite module
    this.model = module;
    TfLiteModel = module;

    //! Loading model buffer
    let modelBufferOffset = this.model._getModelBufferMemoryOffset();

    //! Loading model
    let loadModel = this.model._loadModel(407232);

    //! Init Time
    this.initTime = Date.now() - this.startTime;

    //! Logging Info
    console.log("modeling model");
    console.log('Model buffer memory offset:', modelBufferOffset);
    console.log('Model load result:', loadModel);
    console.log('Input memory offset:',this.model._getInputMemoryOffset());
    console.log('Input height:', this.model._getInputHeight());
    console.log('Input width:', this.model._getInputWidth());
    console.log('Input channels:', this.model._getInputChannelCount());
    console.log('Output memory offset:',this.model._getOutputMemoryOffset());
    console.log('Output height:', this.model._getOutputHeight())
    console.log('Output width:', this.model._getOutputWidth())
    console.log('Output channels:',this.model._getOutputChannelCount());
  }


  //! TFJS model inference
  async runTFJSInference(inputBuf) {

    //! Default return
    let dnnMask = null;

    //! Convert src image from ImageBitMap to tensor type
    const imgData = await tf.browser.fromPixels(inputBuf.data);

    try {

      //! Expand dimension from 3 -> 1 channel
      const expdim = imgData.expandDims(0);

      //! Predict the model output
      var out = await this.model.predict(tf.cast(expdim, 'int32'));
      out = out.argMax(3).expandDims(3);

      //! Threshold the output to obtain mask
      const msk_1 = out.greater(3);
      const msk_2 = out.equal(1);
      const msk_3 = out.equal(2);
      const msk_4 = msk_1.logicalOr(msk_2);
      const msk = msk_4.logicalOr(msk_3);
      dnnMask = tf.cast(msk, 'float32');

      //! Dispose all tensors
      msk_1.dispose();
      msk_2.dispose();
      msk_3.dispose();
      msk_4.dispose();
      msk.dispose();
      expdim.dispose();
      out.dispose();

    } catch(e) {
      console.log("Cannot get alpha mask due to DNN issue " + e);
    }

    //! Return
    return dnnMask;
  }

  //! TFLite model inference
  async runTFLiteInference(inputBuf, isRemoveBG = false) {

    //! Offset variables
    let inputMemoryOffset = this.model._getInputMemoryOffset() / 4;
    let outputMemoryOffset = this.model._getOutputMemoryOffset() / 4;
    let imageData = inputBuf.data;
    let segPixelCount = this.segWidth * this.segHeight;

    //! Assign input data
    for (let i = 0; i < segPixelCount; i++) {
      this.model.HEAPF32[inputMemoryOffset + i * 3] = imageData.data[i * 4] / 255;
      this.model.HEAPF32[inputMemoryOffset + i * 3 + 1] = imageData.data[i * 4 + 1] / 255;
      this.model.HEAPF32[inputMemoryOffset + i * 3 + 2] = imageData.data[i * 4 + 2] / 255;
    }

    //! Run inference
    this.model._runInference();

    //! Get output buffer
    let segmentationMask = new ImageData(this.segWidth, this.segHeight);
    let segmentationMaskOverlay = new ImageData(this.segWidth, this.segHeight);
    for (let i = 0; i < segPixelCount; i++) {
      const background = isRemoveBG
        ? this.model.HEAPF32[outputMemoryOffset + i * 2 + 1]
        : this.model.HEAPF32[outputMemoryOffset + i * 2]
      const person = isRemoveBG
        ? this.model.HEAPF32[outputMemoryOffset + i * 2]
        : this.model.HEAPF32[outputMemoryOffset + i * 2 + 1]

      const shift = Math.max(background, person);
      const backgroundExp = Math.exp(background - shift);
      const personExp = Math.exp(person - shift);
      segmentationMask.data[i * 4 + 3] = (255 * personExp) / (backgroundExp + personExp); // softmax
      if (!isRemoveBG) {
        segmentationMask.data[i * 4 + 2] = (255 * personExp) / (backgroundExp + personExp); // softmax
        segmentationMask.data[i * 4 + 1] = (255 * personExp) / (backgroundExp + personExp); // softmax
        segmentationMask.data[i * 4] = (255 * personExp) / (backgroundExp + personExp); // softmax
      }

      //TODO : Add for UI alpha section
      segmentationMaskOverlay.data[i * 4 + 3] = (255 * personExp) / (backgroundExp + personExp); // softmax

    }

    //! Return mask
    return segmentationMask;
  }

  //! Running ML models with refinement stage
  async run(inputBuf, colorSpace, isRemoveBG = false) {

    //! Return mask
    let dnnMask = null;

    //! Resizing input frame to model size
    const segWidth = this.type === 'tflite' ? 160 : 192
    const segHeight = this.type === 'tflite' ? 96 : 192

    //! Pre-processing input frame
    let srcData = await preProcessing(
      inputBuf.data,
      colorSpace,
      {'width': inputBuf.width, 'height': inputBuf.height},
      ColorSpaceType.COLOR_SPACE_RGBA,
      {'width': segWidth, 'height': segHeight}
    )

    if (srcData === null) {
      return null;
    }

    let input = new ImageData(srcData, segWidth, segHeight);
    let procInputBuf = PsySegBuf(inputBuf.width, inputBuf.height, 4, input);

    //! Check model type
    if (this.type === "tflite") {
      await this.runTFLiteInference(procInputBuf, isRemoveBG)
        .then((mask) => { dnnMask = mask; })
        .catch((e) => { console.log("Cannot get inference due to "+ e); });
    } else {
      await this.runTFJSInference(procInputBuf)
        .then((mask) => { dnnMask = mask; })
        .catch((e) => { console.log("Cannot get inference due to "+ e); });
    }

    //! Return value
    return dnnMask;
  }

  //! Destroy data of SmartEyes
  async release() {
    /*
    if (this.backend !== "tflite") {
        //tf.dispose(this.model);
        delete this.model;
    } else {
        delete this.model;
    }
    */
    delete this.model;
    delete this.segHeight;
    delete this.segWidth;
    delete this.type;
    delete this.runTFJSInference;
    delete this.runTFLiteInference;
    delete this.loadTfLiteModel;
    delete this.loadTfJsModel;
    delete this.run;
  }
}