//! Tensorflow JS library
import * as tf from '@tensorflow/tfjs';

//! Strict mode
"use strict";

//! Preprocessing input to appropriate size / colorspace / type
export async function preProcessing(srcImage, srcColorSpace, srcSize, dstColorSpace, dstSize) {

    //! Output of each pre-processing step
    let colorspaceOut = null;
    let resizeOut = null;

    //! Converting colorspace
    await convertColor(srcImage, srcColorSpace, dstColorSpace)
    .then((processedImg) => { colorspaceOut = processedImg })
    .catch((e) => { 
        console.log("Cannot covert colorspace due to " + e);
    });

    //! Resize src image
    if (colorspaceOut !== null) {
        await convertSize(colorspaceOut, srcSize, dstSize)
        .then((processedImg) => { 
            resizeOut = processedImg; 
        })
        .catch((e) => { 
            console.log("Cannot convert size due to " + e);
        });
    }

    //! Return final output
    return resizeOut;
}

/**
* Convert color between colorspace
*
* @param srcImage The input color buffer.
* @param srcColorSpace The input colorspace
* @param dstColorSpace The output colorspace
*
* @return dstImage after changing colorspace
*/
async function convertColor(srcImage, srcColorSpace, dstColorSpace) {
    try {
        let dstImage = (srcColorSpace !== dstColorSpace) ? srcImage : srcImage;
        return dstImage;
    } catch (e) {
        return null;
    }
}

/**
* Convert size between input and output
*
* @param srcImage The input color buffer.
* @param srcSize The input size
* @param dstSize The output size
*
* @return dstImage after changing size
*/
async function convertSize(srcImage, srcSize, dstSize) {
    
    try {
        
        if ((srcSize.width === dstSize.width) && (srcSize.height === dstSize.height)) {
            return null;
        }

        //! Get input data
        const srcData = srcImage.data;

        //! Create buffer size for input
        const inputSize = Module._get_buffer_size(srcSize.width, srcSize.height);
        const inputMem = Module._create_buffer(inputSize);
        Module.HEAP8.set(srcData, inputMem);

        //! Create buffer size for output
        const outputSize = Module._get_buffer_size(dstSize.width, dstSize.height);
        const outputMem = Module._create_buffer(outputSize);

        //! Resize if necessary
        Module._convertSize(inputMem, outputMem, srcSize.width, srcSize.height, dstSize.width, dstSize.height);

        //! Get resized image buffer
        const dstData = new Uint8ClampedArray(new Uint8Array(Module.HEAP8.buffer, outputMem, outputSize));

        //! Destroy buffer
        Module._destroy_buffer(inputMem);
        Module._destroy_buffer(outputMem);

        //! Return resized image
        return dstData;

        // //! Convert src image from ImageBitMap to tensor type
        // const bim = await tf.browser.fromPixels(srcImage);
        //
        // //! Converted output
        // let dstImage;
        //
        // //! Check difference in image size
        // if ((srcSize.width !== dstSize.width) || (srcSize.height !== dstSize.height)) {
        //
        //     //! Resize & Scaling Image from [0..255] to range [0..1]
        //     dstImage = tf.tidy(() => {return tf.image.resizeBilinear(bim, [dstSize.height, dstSize.width]).div(tf.scalar(255.0))});
        //
        // } else {
        //
        //     //! Only scaling from [0..255] to range [0..1]
        //     dstImage = tf.tidy(() => {return bim.div(tf.scalar(255.0))});
        // }
        //
        // //! Dispose bg image
        // bim.dispose();
        //
        // //! Return converted output
        // return tf.browser.toPixels(dstImage);

    } catch (e) {

        //! Return null
        return null;
    }
    
}