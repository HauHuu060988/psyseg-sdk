const _0x1f0c97=_0x4565;function _0xd9aa(){const _0x5f2476=['model','Cannot\x20get\x20alpha\x20mask\x20due\x20to\x20DNN\x20issue\x20','int32','tflite_model','startTime','30642156joJAcF','max','Input\x20height:','then','2129876kQcPSo','data','equal','_getInputMemoryOffset','segHeight','predict','bind','_getOutputHeight','_getInputChannelCount','_getOutputChannelCount','getBackend','Model\x20buffer\x20memory\x20offset:','3pbVRhm','1CIIinv','tfjs','backend','use\x20strict','run','catch','Input\x20memory\x20offset:','width','default','runTFLiteInference','_loadModel','dispose','_getInputHeight','https://psyjs-cdn.nuvixa.com/model.json','logicalOr','60aqcYqG','cast','Model\x20load\x20result:','1626126RNkxrB','modeling\x20model','10892187qADuhC','now','tflite','height','initTime','_getOutputMemoryOffset','log','runTFJSInference','155936eFZyzX','segWidth','setBackend','argMax','fromPixels','GET','loadGraphModel','Create\x20tflite\x20module\x20failed\x20due\x20to\x20','exp','1406564fkctDz','release','float32','6pAicJo','type','Output\x20memory\x20offset:','webgl','Cannot\x20get\x20inference\x20due\x20to\x20','expandDims','8vXzipj','Loading\x20model\x20error\x20due\x20to\x20','HEAPF32','171200FHMTBr'];_0xd9aa=function(){return _0x5f2476;};return _0xd9aa();}(function(_0x370670,_0x1b5bf1){const _0x2b7a29=_0x4565,_0x37e8e8=_0x370670();while(!![]){try{const _0x319aa3=parseInt(_0x2b7a29(0x183))/0x1*(-parseInt(_0x2b7a29(0x14d))/0x2)+-parseInt(_0x2b7a29(0x182))/0x3*(-parseInt(_0x2b7a29(0x160))/0x4)+-parseInt(_0x2b7a29(0x16c))/0x5*(-parseInt(_0x2b7a29(0x163))/0x6)+-parseInt(_0x2b7a29(0x176))/0x7+parseInt(_0x2b7a29(0x169))/0x8*(-parseInt(_0x2b7a29(0x14f))/0x9)+-parseInt(_0x2b7a29(0x14a))/0xa*(-parseInt(_0x2b7a29(0x157))/0xb)+parseInt(_0x2b7a29(0x172))/0xc;if(_0x319aa3===_0x1b5bf1)break;else _0x37e8e8['push'](_0x37e8e8['shift']());}catch(_0x5e37f3){_0x37e8e8['push'](_0x37e8e8['shift']());}}}(_0xd9aa,0xaa22c));import*as _0xf1497b from'@tensorflow/tfjs';import{PsySegBuf,ColorSpaceType}from'../PsySeg.mjs';import{preProcessing}from'../Preprocessing/Preprocessing.js';_0x1f0c97(0x186);function _0x4565(_0x15cd6e,_0x341873){const _0xd9aa12=_0xd9aa();return _0x4565=function(_0x4565ad,_0xb8270a){_0x4565ad=_0x4565ad-0x14a;let _0xf708ba=_0xd9aa12[_0x4565ad];return _0xf708ba;},_0x4565(_0x15cd6e,_0x341873);}export class SmartEyes{constructor(_0x5cbf10){const _0x202da0=_0x1f0c97;this[_0x202da0(0x171)]=Date['now'](),this[_0x202da0(0x153)]=0x0,this[_0x202da0(0x16d)]=null,this[_0x202da0(0x170)]=null,this[_0x202da0(0x164)]=_0x202da0(0x151),this[_0x202da0(0x158)]=0xa0,this[_0x202da0(0x17a)]=0x60;if(_0x5cbf10===_0x202da0(0x151))this[_0x202da0(0x164)]=_0x202da0(0x151),this[_0x202da0(0x158)]=0xa0,this[_0x202da0(0x17a)]=0x60,createTFLiteModule()[_0x202da0(0x175)](_0x376548=>{const _0x22f486=_0x202da0;this[_0x22f486(0x16d)]=_0x376548;let _0x234c94=this[_0x22f486(0x16d)]['_getModelBufferMemoryOffset'](),_0x20548b=this[_0x22f486(0x16d)][_0x22f486(0x18d)](0x636c0);this[_0x22f486(0x153)]=Date[_0x22f486(0x150)]()-this[_0x22f486(0x171)],console[_0x22f486(0x155)](_0x22f486(0x14e)),console['log'](_0x22f486(0x181),_0x234c94),console['log'](_0x22f486(0x14c),_0x20548b),console[_0x22f486(0x155)](_0x22f486(0x189),this['model']['_getInputMemoryOffset']()),console['log'](_0x22f486(0x174),this[_0x22f486(0x16d)][_0x22f486(0x18f)]()),console['log']('Input\x20width:',this[_0x22f486(0x16d)]['_getInputWidth']()),console[_0x22f486(0x155)]('Input\x20channels:',this[_0x22f486(0x16d)][_0x22f486(0x17e)]()),console['log'](_0x22f486(0x165),this[_0x22f486(0x16d)][_0x22f486(0x154)]()),console[_0x22f486(0x155)]('Output\x20height:',this[_0x22f486(0x16d)][_0x22f486(0x17d)]()),console[_0x22f486(0x155)]('Output\x20width:',this[_0x22f486(0x16d)]['_getOutputWidth']()),console[_0x22f486(0x155)]('Output\x20channels:',this[_0x22f486(0x16d)][_0x22f486(0x17f)]());})[_0x202da0(0x188)](_0x1b7f21=>{const _0x5a5074=_0x202da0;this[_0x5a5074(0x153)]=Date[_0x5a5074(0x150)]()-this[_0x5a5074(0x171)],console['log'](_0x5a5074(0x15e)+_0x1b7f21);});else{this[_0x202da0(0x164)]=_0x202da0(0x184);let _0x36b5fe=_0x202da0(0x190);this[_0x202da0(0x158)]=0xc0,this[_0x202da0(0x17a)]=0xc0;let _0x5d4f07=new Headers(),_0x1024f2={'method':_0x202da0(0x15c),'headers':_0x5d4f07,'mode':'cors','cache':_0x202da0(0x18b)};_0xf1497b[_0x202da0(0x15d)](_0x36b5fe,{'requestInit':_0x1024f2})[_0x202da0(0x175)](_0x4f9d05=>{const _0x1e8093=_0x202da0;_0xf1497b[_0x1e8093(0x159)](_0x1e8093(0x166)),this['model']=_0x4f9d05,console['log']('Current\x20backend\x20is:\x20'+_0xf1497b[_0x1e8093(0x180)]()),this['initTime']=Date[_0x1e8093(0x150)]()-this[_0x1e8093(0x171)];})[_0x202da0(0x188)](_0x35c55b=>{const _0x143793=_0x202da0;this[_0x143793(0x153)]=Date[_0x143793(0x150)]()-this[_0x143793(0x171)],console[_0x143793(0x155)](_0x143793(0x16a)+_0x35c55b);});}this['runTFJSInference']=this['runTFJSInference']['bind'](this),this[_0x202da0(0x18c)]=this[_0x202da0(0x18c)][_0x202da0(0x17c)](this),this[_0x202da0(0x187)]=this['run'][_0x202da0(0x17c)](this);}async['runTFJSInference'](_0x138f82){const _0x27c84b=_0x1f0c97;let _0x530896=null;const _0x48c8c5=await _0xf1497b['browser'][_0x27c84b(0x15b)](_0x138f82[_0x27c84b(0x177)]);try{const _0x2614b7=_0x48c8c5[_0x27c84b(0x168)](0x0);var _0x1a7353=await this[_0x27c84b(0x16d)][_0x27c84b(0x17b)](_0xf1497b['cast'](_0x2614b7,_0x27c84b(0x16f)));_0x1a7353=_0x1a7353[_0x27c84b(0x15a)](0x3)[_0x27c84b(0x168)](0x3);const _0x1dae15=_0x1a7353['greater'](0x3),_0x470cad=_0x1a7353[_0x27c84b(0x178)](0x1),_0x1fb126=_0x1a7353[_0x27c84b(0x178)](0x2),_0x1ba86a=_0x1dae15['logicalOr'](_0x470cad),_0x424147=_0x1ba86a[_0x27c84b(0x191)](_0x1fb126);_0x530896=_0xf1497b[_0x27c84b(0x14b)](_0x424147,_0x27c84b(0x162)),_0x1dae15[_0x27c84b(0x18e)](),_0x470cad['dispose'](),_0x1fb126[_0x27c84b(0x18e)](),_0x1ba86a[_0x27c84b(0x18e)](),_0x424147[_0x27c84b(0x18e)](),_0x2614b7[_0x27c84b(0x18e)](),_0x1a7353[_0x27c84b(0x18e)]();}catch(_0x581519){console[_0x27c84b(0x155)](_0x27c84b(0x16e)+_0x581519);}return _0x530896;}async[_0x1f0c97(0x18c)](_0x377ae6,_0xd9b48b=![]){const _0x2a35d8=_0x1f0c97;let _0x8762fc=this['model'][_0x2a35d8(0x179)]()/0x4,_0x218670=this[_0x2a35d8(0x16d)]['_getOutputMemoryOffset']()/0x4,_0x1ac4d6=_0x377ae6[_0x2a35d8(0x177)],_0x15fc1a=this[_0x2a35d8(0x158)]*this[_0x2a35d8(0x17a)];for(let _0x445353=0x0;_0x445353<_0x15fc1a;_0x445353++){this[_0x2a35d8(0x16d)]['HEAPF32'][_0x8762fc+_0x445353*0x3]=_0x1ac4d6[_0x2a35d8(0x177)][_0x445353*0x4]/0xff,this[_0x2a35d8(0x16d)]['HEAPF32'][_0x8762fc+_0x445353*0x3+0x1]=_0x1ac4d6[_0x2a35d8(0x177)][_0x445353*0x4+0x1]/0xff,this[_0x2a35d8(0x16d)]['HEAPF32'][_0x8762fc+_0x445353*0x3+0x2]=_0x1ac4d6[_0x2a35d8(0x177)][_0x445353*0x4+0x2]/0xff;}this[_0x2a35d8(0x16d)]['_runInference']();let _0x1ba1d1=new ImageData(this[_0x2a35d8(0x158)],this['segHeight']),_0x238489=new ImageData(this[_0x2a35d8(0x158)],this[_0x2a35d8(0x17a)]);for(let _0x855231=0x0;_0x855231<_0x15fc1a;_0x855231++){const _0x15a7fe=_0xd9b48b?this['model']['HEAPF32'][_0x218670+_0x855231*0x2+0x1]:this[_0x2a35d8(0x16d)]['HEAPF32'][_0x218670+_0x855231*0x2],_0x2c7d38=_0xd9b48b?this[_0x2a35d8(0x16d)]['HEAPF32'][_0x218670+_0x855231*0x2]:this[_0x2a35d8(0x16d)][_0x2a35d8(0x16b)][_0x218670+_0x855231*0x2+0x1],_0x10bc4f=Math[_0x2a35d8(0x173)](_0x15a7fe,_0x2c7d38),_0x187352=Math[_0x2a35d8(0x15f)](_0x15a7fe-_0x10bc4f),_0x521e8f=Math[_0x2a35d8(0x15f)](_0x2c7d38-_0x10bc4f);_0x1ba1d1['data'][_0x855231*0x4+0x3]=0xff*_0x521e8f/(_0x187352+_0x521e8f),!_0xd9b48b&&(_0x1ba1d1[_0x2a35d8(0x177)][_0x855231*0x4+0x2]=0xff*_0x521e8f/(_0x187352+_0x521e8f),_0x1ba1d1['data'][_0x855231*0x4+0x1]=0xff*_0x521e8f/(_0x187352+_0x521e8f),_0x1ba1d1[_0x2a35d8(0x177)][_0x855231*0x4]=0xff*_0x521e8f/(_0x187352+_0x521e8f)),_0x238489[_0x2a35d8(0x177)][_0x855231*0x4+0x3]=0xff*_0x521e8f/(_0x187352+_0x521e8f);}return _0x1ba1d1;}async[_0x1f0c97(0x187)](_0x10bf4f,_0x42768d,_0x451d84=![]){const _0x449ae5=_0x1f0c97;let _0x36eed1=null;const _0x3b7662=this[_0x449ae5(0x164)]==='tflite'?0xa0:0xc0,_0x231bb0=this[_0x449ae5(0x164)]===_0x449ae5(0x151)?0x60:0xc0;let _0x19c672=await preProcessing(_0x10bf4f[_0x449ae5(0x177)],_0x42768d,{'width':_0x10bf4f[_0x449ae5(0x18a)],'height':_0x10bf4f[_0x449ae5(0x152)]},ColorSpaceType['COLOR_SPACE_RGBA'],{'width':_0x3b7662,'height':_0x231bb0});if(_0x19c672===null)return null;let _0xc97fc7=new ImageData(_0x19c672,_0x3b7662,_0x231bb0),_0x2020ce=PsySegBuf(_0x10bf4f[_0x449ae5(0x18a)],_0x10bf4f[_0x449ae5(0x152)],0x4,_0xc97fc7);return this[_0x449ae5(0x164)]===_0x449ae5(0x151)?await this[_0x449ae5(0x18c)](_0x2020ce,_0x451d84)[_0x449ae5(0x175)](_0x10db90=>{_0x36eed1=_0x10db90;})[_0x449ae5(0x188)](_0x3f86ba=>{const _0x5a4e76=_0x449ae5;console[_0x5a4e76(0x155)](_0x5a4e76(0x167)+_0x3f86ba);}):await this[_0x449ae5(0x156)](_0x2020ce)[_0x449ae5(0x175)](_0x423933=>{_0x36eed1=_0x423933;})['catch'](_0x255d0c=>{const _0x264657=_0x449ae5;console['log'](_0x264657(0x167)+_0x255d0c);}),_0x36eed1;}async[_0x1f0c97(0x161)](){const _0x2ac841=_0x1f0c97;this[_0x2ac841(0x185)]!==_0x2ac841(0x151)?_0xf1497b[_0x2ac841(0x18e)](this[_0x2ac841(0x16d)]):delete this[_0x2ac841(0x16d)],delete this['segHeight'],delete this[_0x2ac841(0x158)],delete this[_0x2ac841(0x164)],delete this[_0x2ac841(0x156)],delete this[_0x2ac841(0x18c)],delete this[_0x2ac841(0x187)];}}