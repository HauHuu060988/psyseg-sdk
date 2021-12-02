/**
 * Interfaces
 */
export interface IPsySegSetupInfo {
  colorWidth: number
  colorHeight: number
}

export interface IImageSize {
  width: number
  height: number
}

export interface IPsySegBuf {
  width: number,
  height: number,
  channels: number,
  data: ImageData | null
}

export interface IColorSpaceType {
  COLOR_SPACE_RGBA: number
  COLOR_SPACE_BGR: number
  COLOR_SPACE_RGB: number
  COLOR_SPACE_NV21: number
  COLOR_SPACE_NV12: number
  COLOR_SPACE_I420: number
  COLOR_SPACE_YUY2: number
}

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

export interface IPsySegExtraParams {
  erode: number
}

/**
 * Functions
 */
export function ColorSpaceType(): IColorSpaceType

export function PsySegSetupInfo(
  width: number,
  height: number
): IPsySegSetupInfo

export function PsySegExtraParams(
  erode: number
): IPsySegExtraParams

export function PsySegBuf(
  width: number,
  height: number,
  channels: number,
  data: ImageData | null
): IPsySegBuf

export function psy_seg_create(
  pSetupInfo: IPsySegSetupInfo,
  reload?: boolean
): Promise<IPsySegClass>

export function psy_seg_create_internal(
  pSetupInfo: IPsySegSetupInfo,
  backend: string,
  reload?: boolean
): Promise<IPsySegClass>

export function psy_seg_destroy(
  pPsySeg: IPsySegClass | null
): void

export function psy_seg_create_background(
  bgData: any,
  colorSpace: number,
  bgSize: IImageSize,
  dstSize: IImageSize,
): Promise<IPsySegBuf>

export function psy_seg_get_alpha(
  pPsySeg: IPsySegClass | null,
  pInColor: IPsySegBuf | null,
  colorSpace: number,
  pOutAlpha: IPsySegBuf | null
): Promise<boolean>

export function psy_seg_overlay_background(
  pPsySeg: IPsySegClass | null,
  pInColor: IPsySegBuf | null,
  pInBackground: any,
  colorSpace: number,
  pOutColor: IPsySegBuf | null,
  pPsySegExtraParams?: IPsySegExtraParams
): Promise<boolean>

export function psy_seg_overlay_background_new(
  pPsySeg: IPsySegClass | null,
  pInColor: IPsySegBuf | null,
  pInBackground: any,
  colorSpace: number,
  pOutColor: IPsySegBuf | null,
  pPsySegExtraParams?: IPsySegExtraParams
): Promise<boolean>

export function psy_seg_blur_background(
  pPsySeg: IPsySegClass | null,
  pInColor: IPsySegBuf | null,
  colorSpace: number,
  pOutColor: IPsySegBuf | null,
  blurSize: number,
  pPsySegExtraParams?: IPsySegExtraParams
): Promise<boolean>

export function psy_seg_remove_background(
  pPsySeg: IPsySegClass | null,
  pInColor: IPsySegBuf | null,
  colorSpace: number,
  pOutColor: IPsySegBuf | null,
  pPsySegExtraParams?: IPsySegExtraParams
): Promise<boolean>