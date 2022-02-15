//! Tensorflow JS library
import * as tf from '@tensorflow/tfjs';

//! Strict mode
"use strict";

//! Perform mask feathering (Gaussian-blurring + Egde-smoothing)
export function refine(mask, modelSize) {
    
    //! Refinement output
    const refine_out = tf.tidy(() => {
        
        //! Reshape input
        const newmask = mask.slice([0, 0, 0, 0], [1, modelSize.width, modelSize.height, 1]).reshape([1, modelSize.width, modelSize.height, 1]);
        
        //! Gaussian kernel of size (7,7)
        const kernel = tf.tensor4d([0.00092991, 0.00223073, 0.00416755, 0.00606375, 0.00687113, 0.00606375,
            0.00416755, 0.00223073, 0.00535124, 0.00999743, 0.01454618, 0.01648298,
            0.01454618, 0.00999743, 0.00416755, 0.00999743, 0.01867766, 0.02717584,
            0.03079426, 0.02717584, 0.01867766, 0.00606375, 0.01454618, 0.02717584,
            0.03954061, 0.04480539, 0.03954061, 0.02717584, 0.00687113, 0.01648298,
            0.03079426, 0.04480539, 0.05077116, 0.04480539, 0.03079426, 0.00606375,
            0.01454618, 0.02717584, 0.03954061, 0.04480539, 0.03954061, 0.02717584,
            0.00416755, 0.00999743, 0.01867766, 0.02717584, 0.03079426, 0.02717584,
            0.01867766], [7, 7, 1, 1]);
            
        //! Convolve the mask with kernel   
        const blurred = tf.conv2d(newmask, kernel, [1, 1], 'same');
        
        //! Reshape the output
        const fb = blurred.squeeze(0);
        
        //! Normalize the mask  to 0..1 range
        const norm_msk = fb.sub(fb.min()).div(fb.max().sub(fb.min()));
        
        //! Return the result
        return smoothstep(norm_msk);
    });
    
    //! Return refinement output
    return refine_out;
}
  
//! Smooth the mask edges
function smoothstep(x) {
    
    //! Smoothing output
    const smooth_out = tf.tidy(() => {
        
        //! Define the left and right edges
        const edge0 = tf.scalar(0.3);
        const edge1 = tf.scalar(0.5);
        
        //! Scale, bias and saturate x to 0..1 range
        const z = tf.clipByValue(x.sub(edge0).div(edge1.sub(edge0)), 0.0, 1.0);
        
        //! Evaluate polynomial  z * z * (3 - 2 * x)
        return tf.square(z).mul(tf.scalar(3).sub(z.mul(tf.scalar(2))));
    });
    
    // Return smoothing output
    return smooth_out;
}