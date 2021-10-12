// Tensorflow JS library
import * as tf from '@tensorflow/tfjs';

//! Dependencies function
import { MergeBlending, psy_seg_get_alpha_internal } from './Common.js';

/**
 * Overlay background effects with GPU backend
 *
 * @param pPsySeg the PsySeg object
 *
 * @param pInColor The color portion of the input.  We expect this
 * to be an unsigned char buffer with width and height propety corresponding 
 * to SetupData::colorWidth and SetupData::colorHeight.
 * The real size of this buffer will depend on its COLOR_SPACE
 * COLOR_SPACE_BGR/RGB : 3 * width * height bytes
 * COLOR_SPACE_NV21/NV12/I420: width * height + width * height / 2 bytes
 * 
 * @param pInBackground The color portion of the input. We expect this
 * to be an unsigned char buffer with width and height propety corresponding 
 * to SetupData::colorWidth and SetupData::colorHeight.
 * The real size of this buffer will depend on its COLOR_SPACE
 * COLOR_SPACE_BGR/RGB : 3 * width * height bytes
 * COLOR_SPACE_NV21/NV12/I420: width * height + width * height / 2 bytes
 * 
 * @param colorSpace The color space that describes the background color image
 * support COLOR_SPACE_BGR, COLOR_SPACE_RGB, COLOR_SPACE_NV21, COLOR_SPACE_NV12
 * COLOR_SPACE_I420
 * 
 * @param pOutAlpha The alpha mask, value is in range [0, 1].
 * We expect that the that this to
 * be a 1 bytes_per_pixel buffer dimensions corresponding to
 * SetupData::colorWidth and SetupData::colorHeight. The data
 * pointer should point to an appropriately sized allocated array.
 * 
 * @param pOutColor The overlay background buffer with same size as input buffer
 * be a 1 bytes_per_pixel buffer dimensions corresponding to
 * SetupData::colorWidth and SetupData::colorHeight. The data
 * pointer should point to an appropriately sized allocated array.
 * 
 * @param pPsySegExtraParams advanced configuration for customer usages
 * 
 ** @return true on success, false otherise
 *
 */
export async function OverlayBackgroundGPU(pPsySeg, pInColor, pInBackground, colorSpace, pOutAlpha, pOutColor, pPsySegExtraParams = null) {

	//! Overlay BG status
	let status = false;
	let fgImg = null;
	let bgImg = null;
	let mask = null;

	try {

		//! Capture the frame from the webcam.
		if ((pInColor.data !== null) && (pInBackground.data !== null)) {

			//! Getting alpha mask first
			await psy_seg_get_alpha_internal(pPsySeg, pInColor, colorSpace, pOutAlpha)
			.then(async (alpha_status) => {

				//! Check returned status
				if (alpha_status) {

					//! Convert to tensor type
					fgImg = await tf.browser.fromPixels(pInColor.data).div(tf.scalar(255.0));
					bgImg = await tf.browser.fromPixels(pInBackground.data).div(tf.scalar(255.0));
					mask = await tf.browser.fromPixels(pOutAlpha.data).div(tf.scalar(255.0));

					//! Blending background
                    const blend_out =  MergeBlending(
                        fgImg,
                        bgImg,
                        mask
                    );

                    //! Get ImageData in type Uint8ClampedArray
					let convert = await tf.browser.toPixels(blend_out);
					pOutColor.data = new ImageData(convert, pInBackground.width, pInBackground.height);
                    status = true;
					
				} else {
                    console.log("Cannot get alpha mask without error notification");
                }
			})
			.catch((e) => { console.log("cannot get alpha mask due to " + e); });
		} else {
			console.log("Input has a problem, please re-check");
		}
	} catch (e) {
		console.log("Cannot get alpha mask due to " + e);
	}

	// Return status
	return status;
}


/**
 * Overlay background effects with WASM backend
 *
 * @param pPsySeg the PsySeg object
 *
 * @param pInColor The color portion of the input.  We expect this
 * to be an unsigned char buffer with width and height propety corresponding 
 * to SetupData::colorWidth and SetupData::colorHeight.
 * The real size of this buffer will depend on its COLOR_SPACE
 * COLOR_SPACE_BGR/RGB : 3 * width * height bytes
 * COLOR_SPACE_NV21/NV12/I420: width * height + width * height / 2 bytes
 * 
 * @param pInBackground The color portion of the input. We expect this
 * to be an unsigned char buffer with width and height propety corresponding 
 * to SetupData::colorWidth and SetupData::colorHeight.
 * The real size of this buffer will depend on its COLOR_SPACE
 * COLOR_SPACE_BGR/RGB : 3 * width * height bytes
 * COLOR_SPACE_NV21/NV12/I420: width * height + width * height / 2 bytes
 * 
 * @param colorSpace The color space that describes the background color image
 * support COLOR_SPACE_BGR, COLOR_SPACE_RGB, COLOR_SPACE_NV21, COLOR_SPACE_NV12
 * COLOR_SPACE_I420
 * 
 * @param pOutAlpha The alpha mask, value is in range [0, 1].
 * We expect that the that this to
 * be a 1 bytes_per_pixel buffer dimensions corresponding to
 * SetupData::colorWidth and SetupData::colorHeight. The data
 * pointer should point to an appropriately sized allocated array.
 * 
 * @param pOutColor The overlay background buffer with same size as input buffer
 * be a 1 bytes_per_pixel buffer dimensions corresponding to
 * SetupData::colorWidth and SetupData::colorHeight. The data
 * pointer should point to an appropriately sized allocated array.
 * 
 * @param pPsySegExtraParams advanced configuration for customer usages
 * 
 ** @return true on success, false otherise
 *
 */
export async function OverlayBackgroundWASM(pPsySeg, pInColor, pInBackground, colorSpace, pOutAlpha, pOutColor, type, pPsySegExtraParams) {
	
	//! Overlay BG status
	let status = false;
	let bg_image;
	let alpha_image;
	let input_image;
	let memory = null;
	let memorySize = null;
	let bgiMemory = null;
	let bgiMemorySize = null;
	let alphaMemory = null;
	let alphaMemorySize = null;

	try {

		//! Capture the frame from the webcam.
		if ((pInColor.data !== null) && (pInBackground.data !== null)) {

			//! Getting alpha mask first
			await psy_seg_get_alpha_internal(pPsySeg, pInColor, colorSpace, pOutAlpha)
			.then(async (alpha_status) => {

				//! Check returned status
				if (alpha_status) {

						status = true;

						//! Get default parameter for erode value
						let erodeValue = 1;
						if (pPsySegExtraParams !== null) {
							if (pPsySegExtraParams.erode >= 1) {
								erodeValue = 2 * pPsySegExtraParams.erode - 1;
							}
						}

						//! Input frame
						input_image = pInColor.data.data;

						// //! Alpha mask
						// let alphaSrc = await tf.browser.fromPixels(pOutAlpha.data);
						// let alphaData = tf.tidy(() => {return tf.image.resizeBilinear(alphaSrc, [pInColor.height, pInColor.width]).div(tf.scalar(255.0))});
						// alpha_image = await tf.browser.toPixels(alphaData);
						// alphaData.dispose();
						// alphaSrc.dispose();

						//! Background image
						bg_image = pInBackground.data.data;
						alpha_image = pOutAlpha.data.data;

						//! Create buffer for input
						if (!memory) {
							memorySize = Module._get_buffer_size(pInColor.width, pInColor.height);
							memory = Module._create_buffer(memorySize);
						}

						//! Create buffer for background
						if (!bgiMemory) {
							bgiMemorySize = Module._get_buffer_size(pInBackground.width, pInBackground.height);
							bgiMemory = Module._create_buffer(bgiMemorySize);
						}

						//! Create buffer for alpha mask
						if (!alphaMemory) {
							alphaMemorySize = Module._get_buffer_size(pOutAlpha.width, pOutAlpha.height);
							alphaMemory = Module._create_buffer(alphaMemorySize);
						}

						//! Assign data for buffer
						Module.HEAP8.set(input_image, memory);
						Module.HEAP8.set(bg_image, bgiMemory);
						Module.HEAP8.set(alpha_image, alphaMemory);
						
						//! Check type of model
						if (type === "tflite") {
							Module._replaceBackgroundLite(memory, bgiMemory, alphaMemory, pInColor.width, pInColor.height, pOutAlpha.width, pOutAlpha.height, erodeValue);
						} else {
							Module._replaceBackground(memory, bgiMemory, alphaMemory, pInColor.width, pInColor.height);
						}

						//! Convert data array to ImageData
						let imageData = new Uint8ClampedArray(new Uint8Array(Module.HEAP8.buffer, memory, memorySize));
						let image = new ImageData(imageData, pInColor.width, pInColor.height);
                        pOutColor.data = image;

						// memory management
                        //delete alpha_image;
						// alpha_image.delete();
						// input_image.delete();
						// bg_image.delete();
						Module._destroy_buffer(memory);
						Module._destroy_buffer(bgiMemory);
						Module._destroy_buffer(alphaMemory);

					} else {
						console.log("Cannot get alpha mask without error notification");
					}
				})
			.catch((e) => { console.log("cannot get alpha mask due to " + e); });
		} else {
			console.log("Input has a problem, please re-check");
		}
	} catch (e) {
		console.log("Cannot get alpha mask due to " + e);
	}

	// Return status
	return status;
}