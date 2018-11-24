const tf = require('@tensorflow/tfjs');
const irisTrain = require('./training.json');
const irisTest = require('./testing.json');

// convert data
// mapping the test data
const trainingData = tf.tensor2d(irisTrain.map(item => [
    item.sepal_length, item.sepal_width, item.petal_length, item.petal_width
    ]
), [130, 4]);

// mappping the train data
const testingData = tf.tensor2d(irisTest.map(item => [
    item.sepal_length, item.sepal_width, item.petal_length, item.petal_width
    ]
), [14, 4]);

// Output data
const outputData = tf.tensor2d(irisTrain.map(item => [
    item.species === 'setosa' ? 1 : 0,
    item.species === 'viriginca' ? 1 : 0,
    item.species === 'versicolor' ? 1 : 0
]), [130, 3]);

// Creating model
const model = tf.sequential();

model.add(tf.layers.dense({
    inputShape: [4],
    activation: "sigmoid",
    units: 10
}))

model.add(tf.layers.dense({
    inputShape: [4],
    activation: "softmax",
    units: 10
}))



// Compiling model
model.compile({
    loss: "categoricalCrossEntropy",
    optimizer: tf.train.adam

})

// Fitting and predicting model
