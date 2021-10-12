import * as tf from '@tensorflow/tfjs';

export const callInteger1 =()=>{
	tf.engine().startScope();
	console.log('callInteger1')
	tf.engine().endScope();
	return 100

}

export const callString1 =()=>{
	return 'Hello world'
}