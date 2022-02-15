/**
 * Interfaces
 */
/** PsySeg interface for setup information.
 *
 * Currently, it has only size of processing frame. 
 * For further analysis, it may have additional information 
 * that is required for initialization.
 *
 * Notice: processing size is different with
 *         input size. As an example, input size could
 *         be 1080 x 720, but processing size is only
 *         480 x 360. It is a trade-off between quality
 *         and performance. Larger size, better quality
 *         but lower performance (FPS, memory ...)
 *
 */
export interface IPsySegSetupInfo {
  colorWidth: number
  colorHeight: number
}

/** PsySeg interface for size of input / output frame
 *
 * This interface is separated with setup information
 * because of different purpose. This interface is
 * used only for determining image size wheareas setup
 * information may contains other values in the future
 *
 */
export interface IImageSize {
  width: number
  height: number
}

/** PsySeg interface for data buffer object
 * 
 * Instead of using multiple variables for containing
 * data during processing, PsySeg SDK wraps required
 * data into buffer object for easily handling and
 * tracking status.
 * 
 * Notice: PsySeg APIs works only with buffer object. 
 *         Therefore it is necessary to convert original 
 *         data to buffer object for both input / output.
 *
 */
export interface IPsySegBuf {
  width: number,
  height: number,
  channels: number,
  data: ImageData | null
}

/** PsySeg interface for colorspace definition
 *
 * Notice: this interface is for further development
 *         because currently, PsySeg SDK works only
 *         with RGBA colorspace for input / output.
 *
 */
export interface IColorSpaceType {
  COLOR_SPACE_RGBA: number
  COLOR_SPACE_BGR: number
  COLOR_SPACE_RGB: number
  COLOR_SPACE_NV21: number
  COLOR_SPACE_NV12: number
  COLOR_SPACE_I420: number
  COLOR_SPACE_YUY2: number
}

/** PsySeg interface for processing instance
 *
 * This PsySeg instance is the core of SDK which
 * manages user extraction process via deep learning
 * model. This instance is responsible for extracting
 * alpha mask separating persona and background that
 * will be refined in next stage.
 *
 */
export interface IPsySegClass {
  width: number
  height: number
  backend: string
  canvas: HTMLCanvasElement
  canvasContext: CanvasRenderingContext2D
  fgCanvas: HTMLCanvasElement
  fgCanvasContext: CanvasRenderingContext2D
  SmartEyes: any
  runUE(inputBuf, outputBuf, colorSpace, isRemoveBG: boolean): boolean
}

/** PsySeg interface for advanced configuration
 *
 * Currently, PsySeg SDK allows only one parameter
 * for advanced configuration. "erode" value is used
 * for edge refinement based on manual calibration
 * of users.
 *
 */
export interface IPsySegExtraParams {
  erode: number
  enhanceFrame: boolean
  gamma: number
}

/**
 * Functions
 */
/** Colorspace enumerators
 * 
 * @usage : define colorspace of input frame to automatically convert
 *          to RGBA colorspace because our library only work for RGBA
 *          colorspace. 
 * @notice : this is for further development because till now our 
 *           library only support RGB colorspace
 *
 * @arguments : none
 *
 * @returns colorspace value defined in "IColorSpaceType" type
 * 
 */
export function ColorSpaceType(): IColorSpaceType

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
 * @returns setup information object with type "IPsySegSetupInfo"
 *
 */
export function PsySegSetupInfo(
  width: number,
  height: number
): IPsySegSetupInfo

/** PsySeg Extra Parameters
 * 
 * @usage: advanced configuration for customer
 *
 * @arguments
 * @param erode: erode level for edge smoothing
 *
 * @returns extra parameters object with type "IPsySegExtraParams"
 *
 */
export function PsySegExtraParams(
  erode: number,
  enhanceFrame: boolean,
  gamma: number
): IPsySegExtraParams

/** PsySeg Buffer object
 * 
 * @usage : buffers contains information (size / channels) and image
 *          data of input / intermediate output / final output. Our
 *          library will work only on these PsySeg buffers.
 * 
 * @notice : at the beginning, you should create some PsySeg buffers
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
 *          "channels": ..., "data": ...} (type: IPsySegBuf)
 *
 */
export function PsySegBuf(
  width: number,
  height: number,
  channels: number,
  data: ImageData | null
): IPsySegBuf

/** PsySeg instance creation
 * 
 * @usage : create PsySeg instance for managing buffer data and 
 *          executing segmentation process.
 * 
 * @notice : Whenever this API is called, new PsySeg instance will be 
 *           created and this initial step will take some time
 *           (loading model, create internal buffer ...). 
 *           In case recreating a new instance of PsySeg, ensure that 
 *           old instance is removed first using psy_seg_destroy   
 *           function to avoid memory leaks.
 * 
 * @arguments
 * @param pSetupInfo: + type: IPsySegSetupInfo
 *                    + usage: provide setup information
 * @param psyseg_url: + type: String
 *                    + usage: local license file location
 * @param token: + type: String
 *               + usage: access token for license
 * @param key: + type: String
 *             + usage: access key for license
 * @param reload: + type: Boolean
 *                + usage: requesting to reload DNN model (default = true)
 *                         (true = reload / false = keep previous)
 *  
 * @returns PsySeg instance (internal object) (type: IPsySegClass)
 * 
 */
export function psy_seg_create(
  pSetupInfo: IPsySegSetupInfo | null,
  psyseg_url: string, 
  token: string, 
  key: string,
  reload?: boolean
): Promise<IPsySegClass>

/** PsySeg instance creation for internal use only
 * 
 * @usage : create PsySeg instance for managing buffer data and 
 *          executing segmentation process. The difference between
 *          internal creation and external ones is that external creation
 *          is forced to use the best performance backend ("tflite")
 *          wheareas the internal ones is used for specific configuration
 * 
 * @notice : Whenever this API is called, new PsySeg instance will be 
 *           created and this initial step will take some time
 *           (loading model, create internal buffer ...). 
 *           In case recreating a new instance of PsySeg, ensure that 
 *           old instance is removed first using psy_seg_destroy   
 *           function to avoid memory leaks.
 * 
 * @arguments
 * @param pSetupInfo: + type: IPsySegSetupInfo
 *                    + usage: provide setup information
 * @param psyseg_url: + type: String
 *                    + usage: local license file location
 * @param token: + type: String
 *               + usage: access token for license
 * @param key: + type: String
 *             + usage: access key for license
 * @param backend: + type: String
 *                 + usage: required specific backend for segmentation
 *                          Currently, SDK support "tflite", "wasm" &
 *                          "gpu" backend.
 * @param reload: + type: Boolean
 *                + usage: requesting to reload DNN model (default = true)
 *                         (true = reload / false = keep previous)
 *  
 * @returns PsySeg instance (internal object) (type: IPsySegClass)
 *
 */
export function psy_seg_create_internal(
  pSetupInfo: IPsySegSetupInfo | null,
  psyseg_url: string, 
  token: string, 
  key: string,
  backend: string,
  reload?: boolean
): Promise<IPsySegClass>

/** PsySeg instance destroy
 * 
 * @usage : destroy all related tensors and data buffers and close the 
 *          PsySeg instance
 *
 * @arguments
 * @param pPsySeg: + type: IPsySegClass
 *                 + usage: instance is necessary to be destroyed
 *  
 * @returns none
 *
 */
export function psy_seg_destroy(
  pPsySeg: IPsySegClass | null
): Promise<void>

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
 * @param colorSpace: + type: Number (this is selected in "IColorSpaceType"
 *                            enumerations)
 *                    + usage: describing the colorspace of background
 *                             image which is used to convert to RGBA   
 *                             colorspace
 * @param bgSize: + type: IImageSize
 *                + usage: describing a 2D size (width, height) of 
 *                         input background image
 * @param dstSize: + type: IImageSize
 *                 + usage: describing a 2D size (width, height) of 
 *                          output background image (same as output
 *                          size in setup information)
 * 
 ** @return PsySegBuf buffer contains background data (type: IPsySegBuf)
 *
 */
export function psy_seg_create_background(
  bgData: ImageData | null,
  colorSpace: number,
  bgSize: IImageSize | null,
  dstSize: IImageSize | null,
): Promise<IPsySegBuf>

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
 * @param pPsySeg: + type: IPsySegClass
 *                 + usage: management and execution instance
 * @param pInColor: + type: IPsySegBuf (data attributes type = “ImageData”)
 *                  + usage: input buffer data
 * @param colorSpace: + type: Number (this is selected in "IColorSpaceType"
 *                            enumerations)
 *                    + usage: describing the colorspace of background
 *                             image which is used to convert to RGBA   
 *                             colorspace
 * @param pOutAlpha: + type: IPsySegBuf (data attributes type = “ImageData”)
 *                   + usage: alpha mask buffer contains background & 
 *                            foreground information
 * 
 ** @return true on success, false otherwise
 *
 */
export function psy_seg_get_alpha(
  pPsySeg: IPsySegClass | null,
  pInColor: IPsySegBuf | null,
  colorSpace: number,
  pOutAlpha: IPsySegBuf | null
): Promise<boolean>

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
 * @param pPsySeg: + type: IPsySegClass
 *                 + usage: management and execution instance
 * @param pInColor: + type: IPsySegBuf (data attributes type = “ImageData”)
 *                  + usage: input buffer data
 * @param pInBackground: + type: IPsySegBuf (data attributes type = “ImageData”)
 *                       + usage: background buffer data with specific
 *                                size based on type of backend
 * @param colorSpace: + type: Number (this is selected in "IColorSpaceType"
 *                            enumerations)
 *                    + usage: describing the colorspace of background
 *                             image which is used to convert to RGBA   
 *                             colorspace
 * @param pOutColor: + type: IPsySegBuf (data attributes type = “ImageData”)
 *                   + usage: overlayed background output frame with
 *                            same size & colorspace with input frame.
 * @param pPsySegExtraParams: + type: IPsySegExtraParams
 *                   		  + usage: advanced configuration parameters
 * 
 ** @return true on success, false otherwise
 *
 */
export function psy_seg_overlay_background(
  pPsySeg: IPsySegClass | null,
  pInColor: IPsySegBuf | null,
  pInBackground: IPsySegBuf | null,
  colorSpace: number,
  pOutColor: IPsySegBuf | null,
  pPsySegExtraParams?: IPsySegExtraParams | null
): Promise<boolean>

/** Overlaying background effect with new method
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
 * @param pPsySeg: + type: IPsySegClass
 *                 + usage: management and execution instance
 * @param pInColor: + type: IPsySegBuf (data attributes type = “ImageData”)
 *                  + usage: input buffer data
 * @param pInBackground: + type: IPsySegBuf (data attributes type = “ImageData”)
 *                       + usage: background buffer data with specific
 *                                size based on type of backend
 * @param colorSpace: + type: Number (this is selected in "IColorSpaceType"
 *                            enumerations)
 *                    + usage: describing the colorspace of background
 *                             image which is used to convert to RGBA   
 *                             colorspace
 * @param pOutColor: + type: IPsySegBuf (data attributes type = “ImageData”)
 *                   + usage: overlayed background output frame with
 *                            same size & colorspace with input frame.
 * @param pPsySegExtraParams: + type: IPsySegExtraParams
 *                   		  + usage: advanced configuration parameters
 * 
 ** @return true on success, false otherwise
 *
 */
export function psy_seg_overlay_background_new(
  pPsySeg: IPsySegClass | null,
  pInColor: IPsySegBuf | null,
  pInBackground: IPsySegBuf | null,
  colorSpace: number,
  pOutColor: IPsySegBuf | null,
  pPsySegExtraParams?: IPsySegExtraParams | null
): Promise<boolean>

/** Blurring background effect
 * 
 * @usage : extracting user from background and making a background
 *          blurred to hidden unnecessary object.
 *
 * @notice : this function has one parameter called "blurSize". This
 *           parameter is used to modify blurred level of background.
 *           This parameter is an odd number (3, 5, 7...). The more
 *           higher number the more blurred level, but the performance
 *           (CPU & GPU) will be decreased if this number increase.
 * 
 * @arguments
 * @param pPsySeg: + type: IPsySegClass
 *                 + usage: management and execution instance
 * @param pInColor: + type: IPsySegBuf (data attributes type = “ImageData”)
 *                  + usage: input buffer data
 * @param colorSpace: + type: Number (this is selected in "IColorSpaceType"
 *                            enumerations)
 *                    + usage: describing the colorspace of background
 *                             image which is used to convert to RGBA   
 *                             colorspace
 * @param pOutColor: + type: IPsySegBuf (data attributes type = “ImageData”)
 *                   + usage: blurred background output frame with
 *                            same size & colorspace with input frame.
 * @param blurSize: + type: Number (odd = 3,5,7,... / default = 13)
 *                  + usage: blurred level
 * @param pPsySegExtraParams: + type: IPsySegExtraParams
 *                   		  + usage: advanced configuration parameters
 * 
 ** @return true on success, false otherwise
 *
 */
export function psy_seg_blur_background(
  pPsySeg: IPsySegClass | null,
  pInColor: IPsySegBuf | null,
  colorSpace: number,
  pOutColor: IPsySegBuf | null,
  blurSize: number,
  pPsySegExtraParams?: IPsySegExtraParams | null
): Promise<boolean>

/** Remove background effect
 * 
 * @usage : extracting only user from background
 * 
 * @arguments
 * @param pPsySeg: + type: IPsySegClass
 *                 + usage: management and execution instance
 * @param pInColor: + type: IPsySegBuf (data attributes type = “ImageData”)
 *                  + usage: input buffer data
 * @param colorSpace: + type: Number (this is selected in "IColorSpaceType"
 *                            enumerations)
 *                    + usage: describing the colorspace of background
 *                             image which is used to convert to RGBA   
 *                             colorspace
 * @param pOutColor: + type: IPsySegBuf (data attributes type = “ImageData”)
 *                   + usage: blurred background output frame with
 *                            same size & colorspace with input frame.
 * @param pPsySegExtraParams: + type: IPsySegExtraParams
 *                   		  + usage: advanced configuration parameters
 * 
 ** @return true on success, false otherwise
 *
 */
export function psy_seg_remove_background(
  pPsySeg: IPsySegClass | null,
  pInColor: IPsySegBuf | null,
  colorSpace: number,
  pOutColor: IPsySegBuf | null,
  pPsySegExtraParams?: IPsySegExtraParams | null
): Promise<boolean>