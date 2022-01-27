//! Dependencies function
import { BlurBackgroundGPU, BlurBackgroundWASM } from "./Blur.js";
import { OverlayBackgroundGPU, OverlayBackgroundWASM } from './Overlay.js';
import { RemoveBackgroundGPU, RemoveBackgroundWASM } from './RemoveBG.js';

/**
 * Overlay background effects
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
export async function psy_seg_overlay_background_internal(pPsySeg, pInColor, pInBackground, colorSpace, pOutAlpha, pOutColor, pPsySegExtraParams = null) {

	//! Overlay BG status
	let status = false;

	switch (pPsySeg.backend) {
		case "gpu": {
			await OverlayBackgroundGPU(pPsySeg, pInColor, pInBackground, colorSpace, pOutAlpha, pOutColor, pPsySegExtraParams)
			.then(ret => status = ret)
			.catch(e => {
				//console.log("Cannot overlay background GPU due to " + e)
			});
			break;
		}	
		case "wasm": {
			await OverlayBackgroundWASM(pPsySeg, pInColor, pInBackground, colorSpace, pOutAlpha, pOutColor, pPsySeg.backend, pPsySegExtraParams)
			.then(ret => status = ret)
			.catch(e => {
				//console.log("Cannot overlay background WASM due to " + e)
			});
			break;
		}
		case "tflite": {
			await OverlayBackgroundWASM(pPsySeg, pInColor, pInBackground, colorSpace, pOutAlpha, pOutColor, pPsySeg.backend, pPsySegExtraParams)
			.then(ret => status = ret)
			.catch(e => {
				//console.log("Cannot overlay background WASM due to " + e)
			});
			break;
		}
		default: break;
	}

	// Return status
	return status;
}

/**
 * Blurring background effects
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
 * @param pInBackground The color portion of the input.  We expect this
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
export async function psy_seg_blur_background_internal(pPsySeg, pInColor, colorSpace, pOutAlpha, pOutColor, blurSize, pPsySegExtraParams = null) {

    //! Overlay BG status
	let status = false;
	switch (pPsySeg.backend) {
		case "gpu": {
			await BlurBackgroundGPU(pPsySeg, pInColor, colorSpace, pOutAlpha, pOutColor, blurSize, pPsySegExtraParams)
			.then(ret => status = ret)
			.catch(e => {
				//console.log("Cannot blur background GPU due to " + e)
			});
			break;
		}	
		case "wasm": {
			await BlurBackgroundWASM(pPsySeg, pInColor, colorSpace, pOutAlpha, pOutColor, blurSize, pPsySeg.backend, pPsySegExtraParams)
			.then(ret => status = ret)
			.catch(e => {
				//console.log("Cannot blur background WASM due to " + e)
			});
			break;
		}
		case "tflite": {
			await BlurBackgroundWASM(pPsySeg, pInColor, colorSpace, pOutAlpha, pOutColor, blurSize, pPsySeg.backend, pPsySegExtraParams)
			.then(ret => status = ret)
			.catch(e => {
				//console.log("Cannot blur background WASM due to " + e)
			});
			break;
		}
		default: break;
	}

	// Return status
	return status;
}

/**
 * Get persona only effects
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
export async function psy_seg_remove_background_internal(pPsySeg, pInColor, colorSpace, pOutAlpha, pOutColor, pPsySegExtraParams = null) {
	try {
		switch (pPsySeg.backend) {
			case 'gpu': {
				return await RemoveBackgroundGPU(pPsySeg, pInColor, colorSpace, pOutAlpha, pOutColor, pPsySegExtraParams)
			}
			case 'wasm': {
				return await RemoveBackgroundWASM(pPsySeg, pInColor, colorSpace, pOutAlpha, pOutColor, pPsySeg.backend, pPsySegExtraParams)
			}
			case 'tflite': {
				return await RemoveBackgroundWASM(pPsySeg, pInColor, colorSpace, pOutAlpha, pOutColor, pPsySeg.backend, pPsySegExtraParams)
			}
			default:
				break
		}
	} catch (e) {
		//console.log(`Cannot remove background by ${pPsySeg.backend} due to`, e)
	}
	return false
}