// Tensorflow JS library
import * as tf from '@tensorflow/tfjs';

/**
 * Blending foreground and background base on mask
 *
 * @param fgImg foreground image in range [0..1]
 *
 * @param bgImg background image in range [0..1]
 * 
 * @param mask alpha mask in range [0..1]
 * 
 ** @return output blending image in tensor type
 *
 */
export function MergeBlending(fgImg, bgImg, mask) {

    let blend_out = tf.tidy(() => {
		const img_crop = fgImg.mul(mask);
		const bgd_crop = bgImg.mul(tf.scalar(1.0).sub(mask));
		const result = tf.add(img_crop, bgd_crop);
		return result;
    });

    return blend_out;
}

/**
 * Get the alpha mask of the user
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
 * @param callType The type of alpha mask output
 * default = false = change to Uint8ClampedArray
 * true = retain tensor type
 * 
 * @param isRemoveBG
 * 
 ** @return true on success, false otherise
 *
 */
export async function psy_seg_get_alpha_internal(pPsySeg, pInColor, colorSpace, pOutAlpha, isRemoveBG = false) {

	//! Get alpha mask status
	let status = false;

	try {
		
		//! Capture the frame from the webcam.
		if (pInColor.data !== null) {

			//! Running User Extraction
			await pPsySeg.runUE(pInColor, pOutAlpha, colorSpace, isRemoveBG)
			.then((UE_status) => { status = UE_status; })
			.catch((e) => { console.log("Run User Extraction error with " + e); });

		}
	} catch (e) {
		
		//! logging error
		console.log("Cannot get alpha mask due to " + e);
	}

	//! Return status
	return status;
}