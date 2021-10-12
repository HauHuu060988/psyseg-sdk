import * as tf from '@tensorflow/tfjs';

const callInteger1 =()=>{
	tf.engine().startScope();
	console.log('callInteger1')
	tf.engine().endScope();
	return 100

}

const callString1 =()=>{
	return 'Hello world'
}

exports.callInteger = 100
exports.callString = 'Hello world'