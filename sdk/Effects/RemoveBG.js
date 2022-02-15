//! Dependencies function
import * as tf from '@tensorflow/tfjs';
import { psy_seg_get_alpha_internal } from './Common.js'

/**
 * Getting persona only effect with GPU backend
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
export async function RemoveBackgroundGPU(pPsySeg, pInColor, colorSpace, pOutAlpha, pOutColor, pPsySegExtraParams = null) {
  if (!pInColor.data) {
    console.log('Input has a problem, please re-check');
    return false;
  }

  try {
    //! Getting alpha mask first
    const alphaStatus = await psy_seg_get_alpha_internal(pPsySeg, pInColor, colorSpace, pOutAlpha);

    //! Check returned status
    if (!alphaStatus) {
      console.log('Cannot get alpha mask without error notification');
      return false;
    }
  
    //! Convert to tensor type
    const fgImg = await tf.browser.fromPixels(pInColor.data, 4).div(tf.scalar(255.0));
    const mask = await tf.browser.fromPixels(pOutAlpha.data, 4).div(tf.scalar(255.0));
    const ue = fgImg.mul(mask);
    
    // const imageBuffer = await tf.browser.toPixels(tf.tidy(() => {
    //   const [rgb,] = tf.split(ue, [3, 1], -1)
    //   const condition = ue.sum(-1).reshape([pInColor.width * pInColor.height]).equal(1)
    //   const a = tf.scalar(0).where(condition, tf.scalar(1)).reshape([pInColor.width, pInColor.height, 1])
    //   return tf.concat([rgb, a], -1)
    // }))

    /* Recommended: this approach make performance better above) */
    const imageBuffer = await tf.browser.toPixels(ue);
    const l = imageBuffer.length / 4;
    for (let i = 0; i < l; i++) {
      const r = imageBuffer[i * 4];
      const g = imageBuffer[i * 4 + 1];
      const b = imageBuffer[i * 4 + 2];
      const a = imageBuffer[i * 4 + 3];
      if (a === 255 && r === 0 && g === 0 && b === 0) {
        imageBuffer[i * 4 + 3] = 0;
      }
    }
    
    pOutColor.data = new ImageData(imageBuffer, pInColor.width, pInColor.height);
    
  } catch (e) {
    console.log('Cannot get alpha mask at RemoveBackgroundGPU due to ', e);
    return false;
  }

  // Return status
  return true
}

/**
 * Getting persona only effect with WASM backend
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
export async function RemoveBackgroundWASM(pPsySeg, pInColor, colorSpace, pOutAlpha, pOutColor, type, pPsySegExtraParams = null) {
  if (!pInColor.data) {
    console.log('Input has a problem, please re-check');
    return false;
  }

  try {
    //! Getting alpha mask first
    const isRemoveBG = true;
    const alphaStatus = await psy_seg_get_alpha_internal(pPsySeg, pInColor, colorSpace, pOutAlpha, isRemoveBG);

    //! Check returned status
    if (!alphaStatus) {
      console.log('Cannot get alpha mask without error notification');
      return false;
    }

    //! Get default parameter for erode value
    let erodeValue = 1;
    let enhanceFrame = true;
    let gamma = 0.0;
    if (pPsySegExtraParams !== null) {
      enhanceFrame = pPsySegExtraParams.enhanceFrame;
      if (enhanceFrame) {
        gamma = pPsySegExtraParams.gamma;
      }

      if (pPsySegExtraParams.erode >= 1) {
        erodeValue = 2 * pPsySegExtraParams.erode - 1;
      }
    }

    const { width, height } = pInColor;

    /* Resize pOutAlpha to fit pInColor (Recommended: resize by cpp make performance better) */
    // const alphaData = await tf.browser.toPixels(tf.tidy(() => {
    //   return tf.image.resizeBilinear(
    //     tf.browser.fromPixels(pOutAlpha.data, 4).div(tf.scalar(255.0)),
    //     [height, width]
    //   )
    // }))
    // pOutAlpha.data = new ImageData(alphaData, width, height)

    //! Input frame
    const input_image = pInColor.data.data;
    const alpha_image = pOutAlpha.data.data;

    //! Create buffer for input
    const memorySize = Module._get_buffer_size(width, height);
    const memory = Module._create_buffer(memorySize);

    //! Create buffer for alpha mask
    const alphaMemorySize = Module._get_buffer_size(width, height);
    const alphaMemory = Module._create_buffer(alphaMemorySize);

    //! Assign data for buffer
    Module.HEAP8.set(input_image, memory);
    Module.HEAP8.set(alpha_image, alphaMemory);

    //! Check type of model
    if (type === 'tflite') {
      Module._removeBackgroundLite(memory, alphaMemory, width, height, pOutAlpha.width, pOutAlpha.height, erodeValue, enhanceFrame, gamma);
    } else {
      Module._removeBackground(memory, alphaMemory, width, height, pOutAlpha.width, pOutAlpha.height);
    }

    //! Convert data array to ImageData
    const imageBuffer = new Uint8ClampedArray(new Uint8Array(Module.HEAP8.buffer, memory, memorySize));
    pOutColor.data = new ImageData(imageBuffer, width, height);

    Module._destroy_buffer(memory);
    Module._destroy_buffer(alphaMemory);

  } catch (e) {
    console.log('Cannot get alpha mask at RemoveBackgroundWASM due to ', e);
    return false;
  }

  // Return status
  return true;
}
