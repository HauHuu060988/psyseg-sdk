// Tensorflow JS library
import * as tf from '@tensorflow/tfjs';

// Dependent functions
import { SmartEyes } from "./ML/ML.js";
import { preProcessing } from "./Preprocessing/Preprocessing.js";
import {
	psy_seg_overlay_background_internal,
	psy_seg_blur_background_internal,
	psy_seg_remove_background_internal
} from "./Effects/Effects.js";
import { psy_seg_get_alpha_internal } from './Effects/Common.js';
import { refine } from './Refinement/Refinement.js';

//! Strict mode
"use strict";

/** Colorspace enumerators
 * 
 * @usage : define colorspace of input frame to automatically convert
 *          to RGBA colorspace because our library only work for RGBA
 *          colorspace. 
 * @notice : this is for further development because till now our 
 *           library only support RGB colorspace
 * 
 */
export function ColorSpaceType() {
	return {
		'COLOR_SPACE_RGBA': 0,
		'COLOR_SPACE_BGR': 1,
		'COLOR_SPACE_RGB': 2,
		'COLOR_SPACE_NV21': 3,
		'COLOR_SPACE_NV12': 4,
		'COLOR_SPACE_I420': 5,
		'COLOR_SPACE_YUY2': 6
	}
};

/** PsySeg setup information
 * 
 * @usage : define setup information for PsySeg object including size
 *          of input / output frame in pixels
 * 
 * @notice : our library keeps same size for input and output frames.
 *           Therefore, please scaling input frame to appropriate size 
 *           which is able to resize to another dimension to avoid
 *           re-defining setup information many times.
 *
 * @example : your conference will contains many people and size of
 *            displayed input depends on number of people inside 
 *            conference. Because our segmentation library depends on 
 *            setup information and is unchanged until the end, so
 *            when size of input frame change, we must change setup                 
 *            information and this is not reasonable. We suggest 
 *            providing appropriate size in setup information which is
 *            calculated on average of min - max display input size of 
 *            your conference application. Taking an example, your 
 *            input frame will display in size of 1080 x 720 when
 *            conference has only 2 people, but this size will be 
 *            reduced to 320 x 240 if increasing to 10 people. Instead  
 *            of re-defining setup information 2 times 1080 x 720 to
 *            320 x 240, we suggest defining only 1 time with size
 *            640 x 480. It means that if input is in size 1080 x 720 
 *            or 320 x 240, we must resize it to size of 640 x 480 
 *            before giving to PsySeg library and after processing, we 
 *            continuously resize it from 640 x 480 to 1080 x 720 or 
 *            320 x 240. It takes 2 times for scaling input / output 
 *            but we don't change setup information of PsySeg library 
 *            which will take more processing time.
 * 
 * @arguments
 * @param width: width of input frame in pixels (type: Number)
 * @param height: height of input frame in pixels (type: Number)
 *  
 * @returns setup information object {"width": ... , "height": ...}
 * 
 */
export function PsySegSetupInfo(width, height) {
	return {
		//! Physical height of the image in pixel. it is not the size of the data buffer
		'colorWidth': width,

		//! Physical width of the image in pixel, It is not the size of the data buffer
		'colorHeight': height
	}
};

/** PsySeg Extra Parameters
 * 
 * @usage : advanced configuration for customer
 * 
 * @arguments
 * @param erode: erode level for edge smoothing
 * 
 */ 
export function PsySegExtraParams(erode = 1) {
	return {

		//! erode level for refinements
		'erode': erode
		
	}
}

/** PsySeg Buffer
 * 
 * @usage : buffer contains information (size / channels) and image
 *          data of input / intermediate output / final output. Our
 *          library will work only on PsySeg buffer.
 * 
 * @notice : at the beginning, you should create some PsySeg buffer
 *           for your application to communicate with our library. It 
 *           includes: input buffer, final output buffer, alpha mask 
 *           buffer (optional), background buffer (optional).
 * 
 * @arguments
 * @param width: width of input frame in pixels (type: Number)
 * @param height: height of input frame in pixels (type: Number)
 * @param channels: number of channels of input (ex: colorspace of 
 *                  input is RGBA => it has 4 channels) (type: Number)
 * @param data: image data (type: ImageData)
 *  
 * @returns PsySeg buffer object {"width": ..., "height": ...,  
 *          "channels": ..., "data": ...}
 * 
 */
export function PsySegBuf(width, height, channels, data) {

	return {
		//! Physical width of the image in pixel, It is not the size of the data buffer
		'width': width,

		//! Physical height of the image in pixel, It is not the size of the data buffer
		'height': height,

		//! Number of channels
		'channels': channels,

		//! Holding image data (type: ImageData)
		'data': data
	}
};

/** PsySeg Objects
 * 
 * @param Usage: management and execution object contains attributes 
 *               and methods for getting alpha mask based on DNN model 
 * 
 * @param Notice: PsySeg object is core of library which is responsible
 *                for executing DNN model to extract user from background
 * 
 * @Constructor
 * @param pSetupInfo: + type: PsySegSetupInfo
 *                    + usage: provide setup information
 * @param backend: + type: String
 *                 + usage: type of model backend. Currently, 3 types are
 *                          supported: "tflite" (default), "gpu", "wasm"
 * @param reload: + type: Boolean
 *                + usage: requesting to reload DNN model (default = true) 
 *                         (true = reload / false = keep previous)
 * 
 * @Attributes
 * @param width: width of input frame in pixels
 * @param height: height of input frame in pixels
 * @param SmartEyes: ML instance for executing DNN model
 * @param backend: type of model
 * @param canvas: fake canvas for final output
 * @param canvasContext: context of final output canvas
 * @param fgCanvas: fake canvas for foreground
 * @param fgCanvasContext: context for foreground canvas
 *
 * @Methods
 * @function runUE
 *       1. Usage: extracting user from background based on DNN model
 *       2. Arguments:
 *          a. inputBuf (type: PsySegBuf): input buffer data
 *          b. outputBuf (type: PsySegBuf): output buffer data
 *       3. Return: true => successful segmentation
 *                  false => failed segmentation
 * 
 */
class PsySeg {

	//! Constructor
	constructor(pSetupInfo, backend, reload) {

		//! Size of input / output
		this.width = pSetupInfo.colorWidth;
		this.height = pSetupInfo.colorHeight;

		//! Set backend
		this.backend = backend;

		//! Fake canvas
		this.canvas = document.createElement("canvas");
		this.canvas.width = this.width;
		this.canvas.height = this.height;
		this.canvasContext = this.canvas.getContext('2d');
    
		//! Foreground canvas
		this.fgCanvas = document.createElement("canvas");
		this.fgCanvas.width = this.width;
		this.fgCanvas.height = this.height;
		this.fgCanvasContext = this.fgCanvas.getContext('2d');

		//! SmartEyes for ML implementation
		this.SmartEyes = null;
		this.SmartEyes = new SmartEyes(this.backend, reload);

		//! Bind methods
		this.runUE = this.runUE.bind(this);

	}

	//! User Extraction
	async runUE(inputBuf, outputBuf, colorSpace, isRemoveBG = false) {

		//! User Extraction status
		let status = false;
		let dnnMask = null;

		//! Set default status
		status = false;

		//! Get size of model
		const segWidth = this.backend === 'tflite' ? 160 : 192
		const segHeight = this.backend === 'tflite' ? 96 : 192
    	let modelSize = {'width': segWidth, 'height': segHeight};

		//! Running UE
		await this.SmartEyes.run(inputBuf, colorSpace, isRemoveBG)
		.then(async (output) => {

			//! Check type of backend
			if (this.backend === "tflite") {

				//! Case 1: only assign mask data for output buffer with "tflite" backend
				outputBuf.data = output;
				outputBuf.width = modelSize.width;
				outputBuf.height = modelSize.height;

			} else {

				//! Case 2: refining state for tfjs backend & convert dnn mask to image data
				dnnMask = tf.tidy(() => { return refine(output.div(tf.scalar(255.0)), modelSize).reshape([segHeight, segWidth, 1]); });
				let maskData = await tf.browser.toPixels(dnnMask);
				let maskImageData = new ImageData(maskData, segWidth, segHeight);

				//! Resize alpha mask to input's size
				let finalMask = await preProcessing(
					maskImageData,
					colorSpace,
					{'width': modelSize.width, 'height': modelSize.height},
					ColorSpaceType.COLOR_SPACE_RGBA,
					{'width': inputBuf.width, 'height': inputBuf.height}
				)

				//! Assign mask data for output buffer
				outputBuf.data = new ImageData(finalMask, inputBuf.width, inputBuf.height);
				outputBuf.width = inputBuf.width;
				outputBuf.height = inputBuf.height;

			}
			
			//! Succesful status
			status = true;
		})
		.catch((e) => { console.log("Cannot get alpha mask due to " + e); });

		//! Return status
		return status;
	}
}

/** PsySeg instance creation for customer (the best model)
 * 
 * @usage : create PsySeg instance for managing buffer data and 
 *          executing segmentation process.
 * 
 * @notice : whenever this API is called, new PsySeg instance will be 
 *           created and this initial step will take more time
 *           (loading model, create internal buffer ...). Also, 
 *           ensuring that old instance is removed by psy_seg_destroy   
 *           function to avoid memory leaks.
 * 
 * @arguments
 * @param pSetupInfo: + type: PsySegSetupInfo
 *                    + usage: provide setup information
 * @param reload: + type: Boolean
 *                + usage: requesting to reload DNN model (default = true) 
 *                         (true = reload / false = keep previous)
 * 
 * @returns PsySeg instance (internal object)
 * 
 */
export async function psy_seg_create(pSetupInfo, reload = true) {

	//! Return PsySeg objects
	return new PsySeg(pSetupInfo, "tflite", reload);
}

/** PsySeg instance creation for internal test
 * 
 * @usage : create PsySeg instance for managing buffer data and 
 *          executing segmentation process.
 * 
 * @notice : whenever this API is called, new PsySeg instance will be 
 *           created and this initial step will take more time
 *           (loading model, create internal buffer ...). Also, 
 *           ensuring that old instance is removed by psy_seg_destroy   
 *           function to avoid memory leaks.
 * 
 * @arguments
 * @param pSetupInfo: + type: PsySegSetupInfo
 *                    + usage: provide setup information
 * @param backend: + type: String
 *                 + usage: type of model backend. Currently, 3 types are
 *                          supported: "tflite" (default), "gpu", "wasm"
 * @param reload: + type: Boolean
 *                + usage: requesting to reload DNN model (default = true) 
 *                         (true = reload / false = keep previous)
 *  
 * @returns PsySeg instance (internal object)
 * 
 */
export async function psy_seg_create_internal(pSetupInfo, backend = 'tflite', reload = true) {

	//! Return PsySeg objects
	return new PsySeg(pSetupInfo, backend, reload);
}

/** PsySeg instance destroy
 * 
 * @usage : destroying all related tensor and data buffer inside 
 *          PsySeg instance to avoid memory leaks.
 * 
 * @notice : there still has few garbage memories due to this function
 *           is in development phase and it is not completed to 
 *           destroy all unused memory.
 * 
 * @arguments
 * @param pPsySeg: + type: PsySeg
 *                 + usage: instance is necessary to be destroyed
 *  
 * @returns none
 * 
 */
export async function psy_seg_destroy(pPsySeg) {

	//! Dispose model except "tflite" (automatically destroyed)
	if (pPsySeg.backend !== "tflite") {
		pPsySeg.SmartEyes.release();
	}
	
	//! Delete all method + buffer
	delete pPsySeg.backend;
	delete pPsySeg.width;
	delete pPsySeg.height;
	delete pPsySeg.runUE;
	delete pPsySeg.SmartEyes;
}

/** PsySeg create background image
 * 
 * @usage : pre-processing background image data to reach desired size 
 *          & RGBA colorspace
 *
 * @notice : this function is only necessary for overlay background
 *           effect. Different with input frame, background image
 *           will be re-used frame-by-frame, so it is only necessary
 *           to be converted 1 time at the beginning (or when you
 *           change background) while input frame will be converted
 *           automatically frame-by-frame inside processing function
 * 
 * @arguments
 * @param bgData: + type: ImageData
 *                + usage: background image with type ImageData
 * @param colorSpace: + type: ColorSpaceType
 *                    + usage: describing the colorspace of background
 *                             image which is used to convert to RGBA   
 *                             colorspace
 * @param bgSize: + type: Object
 *                + usage: describing a 2D size (width, height) of 
 *                         input background image
 * @param dstSize: + type: Object
 *                 + usage: describing a 2D size (width, height) of 
 *                          output background image (same as output
 *                          size in setup information)
 * 
 ** @return PsySegBuf buffer contains background image information & data
 *
 */
export async function psy_seg_create_background(bgData, colorSpace, bgSize, dstSize) {
	 
	//! Default background frame
	let processedBG = null;

	//! Pre-processing background buffer
	await preProcessing(
		bgData,
		colorSpace,
		bgSize,
		ColorSpaceType().COLOR_SPACE_RGB,
		dstSize
	)
	.then((outputData) => { processedBG = outputData; })
	.catch((e) => { console.log("Cannot create background due to " + e) });

	//! Return PsySeg buffer
	return PsySegBuf(dstSize.width, dstSize.height, 4, processedBG);
}

/** Get the alpha mask of the user
 * 
 * @usage : extracting user from background and providing alpha mask
 *          for further development
 *
 * @notice : this function is usually used as intermediate step and
 *           need to be combined with effect or beauty step later.
 *           It is used for development phase or advanced usage by
 *           users.
 * 
 * @arguments
 * @param pPsySeg: + type: PsySeg
 *                 + usage: management and execution instance
 * @param pInColor: + type: PsySegBuf (data = "ImageData")
 *                  + usage: input buffer data
 * @param colorSpace: + type: ColorSpaceType
 *                    + usage: describing the colorspace of background
 *                             image which is used to convert to RGBA   
 *                             colorspace
 * @param pOutAlpha: + type: PsySegBuf (data = "ImageData")
 *                   + usage: alpha mask buffer contains background &
 *                            foreground information
 * 
 ** @return true on success, false otherwise
 *
 */
export async function psy_seg_get_alpha(pPsySeg, pInColor, colorSpace, pOutAlpha) {

	//! Get alpha mask status
	let status = false;

	//! Start tf container
	tf.engine().startScope();

	//! Run alpha mask
	if (pPsySeg.SmartEyes.model !== null) {
		await psy_seg_get_alpha_internal(pPsySeg, pInColor, colorSpace, pOutAlpha, true)
		.then((out) => { status = out; })
		.catch((e) => { console.log("Cannot get alpha effects due to " + e); });
	}

	//! Post-processing (only resizing)
	if (pPsySeg.backend === "tflite") {

		//! Resize alpha mask to input's size
		let finalMask = await preProcessing(
			pOutAlpha.data,
			colorSpace,
			{'width': pOutAlpha.width, 'height': pOutAlpha.height},
			ColorSpaceType.COLOR_SPACE_RGBA,
			{'width': pInColor.width, 'height': pInColor.height}
		)

		//! Set value for alpha mask after resizing
		pOutAlpha.data = new ImageData(finalMask, pInColor.width, pInColor.height);
		pOutAlpha.width = pInColor.width;
		pOutAlpha.height = pInColor.height;

	}

	//! End tf container
	tf.engine().endScope();

	// Return status
	return status;
}

/** Overlaying background effect
 * 
 * @usage : extracting user from background and overlaying it with
 *          desired ones.
 *
 * @notice : this function has one special parameter is pInBackground.
 *           This parameter is an output of psy_seg_create_background,
 *           so before calling this function, a background input must
 *           be converted to right template via the above function.
 * 
 * @arguments
 * @param pPsySeg: + type: PsySeg
 *                 + usage: management and execution instance
 * @param pInColor: + type: PsySegBuf (data = "ImageData")
 *                  + usage: input buffer data
 * @param pInBackground: + type: PsySegBuf (data = "ImageData")
 *                       + usage: background buffer data with "specific 
 *                        size" based on type of backend
 * @param colorSpace: + type: ColorSpaceType
 *                    + usage: describing the colorspace of background
 *                             image which is used to convert to RGBA   
 *                             colorspace
 * @param pOutColor: + type: PsySegBuf (data = "ImageData")
 *                   + usage: overlayed background output frame with
 *                            same size & colorspace with input frame.
 * @param pPsySegExtraParams: + type: PsySegExtraParams
 * 							  + usage: advanced configuration parameters
 * 
 ** @return true on success, false otherwise
 *
 */
export async function psy_seg_overlay_background(pPsySeg, pInColor, pInBackground, colorSpace, pOutColor, pPsySegExtraParams = null) {

	//! Overlay BG status
	let status = false;

	//! Start tf container
	tf.engine().startScope();

	//! Default alpha buffer
	//! Resizing input frame to model size
	const segWidth = pPsySeg.backend === 'tflite' ? 160 : 192
	const segHeight = pPsySeg.backend === 'tflite' ? 96 : 192
	let pOutAlpha = PsySegBuf(segWidth, segHeight, 4, null);

	//! Overlay Background
	if (pPsySeg.SmartEyes.model !== null) {
		await psy_seg_overlay_background_internal(pPsySeg, pInColor, pInBackground, colorSpace, pOutAlpha, pOutColor, pPsySegExtraParams)
		.then((out) => { status = out })
		.catch((e) => {});
	}

	// End tf container
	tf.engine().endScope();

	// Return status
	return status;
}

/** Overlaying background effect with new methods
 * 
 * @usage : extracting user from background and overlaying it with
 *          desired ones.
 *
 * @notice : this function has one special parameter is pInBackground.
 *           This parameter is an output of psy_seg_create_background,
 *           so before calling this function, a background input must
 *           be converted to right template via the above function.
 * 
 * @arguments
 * @param pPsySeg: + type: PsySeg
 *                 + usage: management and execution instance
 * @param pInColor: + type: PsySegBuf (data = "ImageData")
 *                  + usage: input buffer data
 * @param pInBackground: + type: PsySegBuf (data = "ImageData")
 *                       + usage: background buffer data with "specific 
 *                        size" based on type of backend
 * @param colorSpace: + type: ColorSpaceType
 *                    + usage: describing the colorspace of background
 *                             image which is used to convert to RGBA   
 *                             colorspace
 * @param pOutColor: + type: PsySegBuf (data = "ImageData")
 *                   + usage: overlayed background output frame with
 *                            same size & colorspace with input frame.
 * @param pPsySegExtraParams: + type: PsySegExtraParams
 * 							  + usage: advanced configuration parameters
 * 
 ** @return true on success, false otherwise
 *
 */
export async function psy_seg_overlay_background_new(pPsySeg, pInColor, pInBackground, colorSpace, pOutColor, pPsySegExtraParams = null) {

	//! Start tf container
	tf.engine().startScope();

	//! Default alpha buffer
	//! Resizing input frame to model size
	const segWidth = pPsySeg.backend === 'tflite' ? 160 : 192
	const segHeight = pPsySeg.backend === 'tflite' ? 96 : 192
	let pOutAlpha = PsySegBuf(segWidth, segHeight, 4, null);

	//! Remove background
	const status = await psy_seg_remove_background(pPsySeg, pInColor, colorSpace, pOutColor, pPsySegExtraParams);

	//! Plot background canvas
	pPsySeg.canvasContext.putImageData(pInBackground.data, 0, 0);
	pPsySeg.fgCanvasContext.clearRect(0, 0, pPsySeg.width, pPsySeg.height);
	pPsySeg.fgCanvasContext.putImageData(pOutColor.data, 0, 0);
	pPsySeg.canvasContext.drawImage(pPsySeg.fgCanvas, 0, 0);

	//! Get output data
	pOutColor.data = pPsySeg.canvasContext.getImageData(0, 0, pPsySeg.width, pPsySeg.height);

	// End tf container
	tf.engine().endScope();

	// Return status
	return status;
}

/** Blurring background effect
 * 
 * @usage : extracting user from background and making a background
 *          blurred to hidden unnecessary object.
 * 
 * @arguments
 * @param pPsySeg: + type: PsySeg
 *                 + usage: management and execution instance
 * @param pInColor: + type: PsySegBuf (data = "ImageData")
 *                  + usage: input buffer data
 * @param colorSpace: + type: ColorSpaceType
 *                    + usage: describing the colorspace of background
 *                             image which is used to convert to RGBA   
 *                             colorspace
 * @param pOutColor: + type: PsySegBuf (data = "ImageData")
 *                   + usage: blurred background output frame with
 *                            same size & colorspace with input frame.
 * @param blurSize: + type: Number (odd = 3,5,7,... / default = 13)
 *                  + usage: blurred level
 * @param pPsySegExtraParams: + type: PsySegExtraParams
 * 							  + usage: advanced configuration parameters
 * 
 ** @return true on success, false otherwise
 *
 */
export async function psy_seg_blur_background(pPsySeg, pInColor, colorSpace, pOutColor, blurSize = 9, pPsySegExtraParams = null) {

	//! Overlay BG status
	let status = false;

	//! Start tf container
	tf.engine().startScope();

	//! Default alpha buffer
	//! Resizing input frame to model size
	const segWidth = pPsySeg.backend === 'tflite' ? 160 : 192;
	const segHeight = pPsySeg.backend === 'tflite' ? 96 : 192;
	let pOutAlpha = PsySegBuf(segWidth, segHeight, 4, null);

	//! Overlay Background
	if (pPsySeg.SmartEyes.model !== null) {
		await psy_seg_blur_background_internal(pPsySeg, pInColor, colorSpace, pOutAlpha, pOutColor, blurSize, pPsySegExtraParams)
		.then((out) => { status = out; })
		.catch((e) => {});
	}

	// End tf container
	tf.engine().endScope();

	// Return status
	return status;
}

/** Remove background effect
 * 
 * @usage : extracting only user from background
 * 
 * @arguments
 * @param pPsySeg: + type: PsySeg
 *                 + usage: management and execution instance
 * @param pInColor: + type: PsySegBuf (data = "ImageData")
 *                  + usage: input buffer data
 * @param colorSpace: + type: ColorSpaceType
 *                    + usage: describing the colorspace of background
 *                             image which is used to convert to RGBA   
 *                             colorspace
 * @param pOutColor: + type: PsySegBuf (data = "ImageData")
 *                   + usage: blurred background output frame with
 *                            same size & colorspace with input frame.
 * @param pPsySegExtraParams: + type: PsySegExtraParams
 * 							  + usage: advanced configuration parameters
 * 
 ** @return true on success, false otherwise
 *
 */
export async function psy_seg_remove_background(pPsySeg, pInColor, colorSpace, pOutColor, pPsySegExtraParams = null) {
	if (!pPsySeg.SmartEyes.model) return

	//! Start tf container
	tf.engine().startScope();

	//! Default alpha buffer
	//! Resizing input frame to model size
	const segWidth = pPsySeg.backend === 'tflite' ? 160 : 192;
	const segHeight = pPsySeg.backend === 'tflite' ? 96 : 192;
	let pOutAlpha = PsySegBuf(segWidth, segHeight, 4, null);

	//! Remove background
	const status = await psy_seg_remove_background_internal(pPsySeg, pInColor, colorSpace, pOutAlpha, pOutColor, pPsySegExtraParams);

	// End tf container
	tf.engine().endScope();

	// Return status
	return status
}