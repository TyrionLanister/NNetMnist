/**
 * Created by TyrionLannister on 23-05-2017.
 */

let async = require('async');
let numeric = require('numeric');
let  trainData = [];
let trainLbl = [];
let  testData = [];
let testLbl = [];
const  layers = [784,30,10];
const epochs = 30;
const batchSize = 10;
const eta = 3.0;
function zerosMatrixFast() {
    this.weights= new Array(2);
    this.biases = new Array(2);
    for (let i = 1, len = layers.length; i < len; i++) {
        this.weights[i-1] = numeric.rep([layers[i-1], layers[i]],0);
        this.biases[i-1]  = numeric.rep([layers[i]],0);
    }
};

function sigmoid(input) {
    return numeric.div(1,numeric.add(1.0, numeric.exp(numeric.mul(-1,input))));
};
function sigmoidDerivative(input) {
    return numeric.mul(sigmoid(input), numeric.sub(1.0,sigmoid(input)));
};
function randomNormFast() {
    return Math.sqrt(-2 * Math.log(Math.random()))*Math.cos((2*Math.PI) * Math.random());
}

function shuffle(data,label){
    for(var j, x, i = data.length; i;
        j = parseInt(Math.random() * i), x = data[--i], data[i] = data[j], data[j] = x,
            x = label[i], label[i] = label[j], label[j] = x);
    return [data,label];
};


function downloadData () {
    let  request = require("request");
    const  fs = require("fs"), zlib = require("zlib");
    async.series([
            function (callback) {

                let stream = request({
                        "uri":"http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
                        "encoding": null,
                    },
                    function (error, response, body) {
                        if (error) {
                            return console.error('download failed:', error);
                        }

                    }
                ).pipe(zlib.createGunzip())
                    .pipe(fs.createWriteStream('train-images-idx3-ubyte'));
                stream.on('finish', function () { callback()});
            },

            function (callback) {
                let stream =  request({
                        "uri":"http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
                        "encoding": null
                    },
                    function (error, response, body) {
                        if (error) {
                            return console.error('download failed:', error);
                        }

                    }
                ).pipe(zlib.createGunzip())
                    .pipe(fs.createWriteStream('train-labels-idx1-ubyte'));
                stream.on('finish', function () { readData()});
            },

        ],
        // optional callback
        function(err, results){
            console.log('error')
        }
    );

}
function  readData() {
    let fs = require('fs');
    let dataFileBuffer = fs.readFileSync('train-images-idx3-ubyte');
    let labelFileBuffer = fs.readFileSync('train-labels-idx1-ubyte');
    for (var image = 0; image < 50000; image++) {
        trainData[image] = new Array(784);
        trainLbl[image] = new Array(10).fill(0);
        for (var row = 0; row <= 27; row++) {
            for (var col = 0; col <= 27; col++) {
                trainData[image][row * 28 + col ] = (dataFileBuffer[(image * 28 * 28) + (row * 28 + col ) + 16]/256);
            }
        }
        trainLbl[image][labelFileBuffer[image + 8]] = 1;
    }

    for (var image = 50000; image < 60000; image++) {
        testData[image-50000] = new Array(784);
        for (var row = 0; row <= 27; row++) {
            for (var col = 0; col <= 27; col++) {
                testData[image-50000][row * 28 + col ] = (dataFileBuffer[(image * 28 * 28) + (row * 28 + col ) + 16]/256);
            }
        }
        testLbl[image-50000] = labelFileBuffer[image + 8];

    }

};
function initializeWeights(){
    this.weights = new Array(2);
    this.biases = new Array(2);
    for (var i = 1, len = layers.length; i < len; i++) {
        this.weights[i-1] = numeric.rep([layers[i-1], layers[i],0]).map(rows => rows.map(randomNormFast));
        this.biases[i-1] = numeric.rep( [1,layers[i]],0).map(rows => rows.map(randomNormFast));
    }
}


function feedforward ( data) {
    for(let i = 0;i <BWts.biases.length;i++){
        data = numeric.add(numeric.dot(data , BWts.weights[i]), replicate(BWts.biases[i],data.length));
        data  = sigmoid(data);
    }
    return data;
}

function replicate(weight, rows) {
    return numeric.dot(numeric.rep([rows,1],1), weight);
}



function backprop(data, label, temp) {
    let activations = new Array(layers.length), zs = new Array(layers.length - 1);
    activations[0] = data;
    let z;
    let Len = data.length;
    //Forward pass
    for (let i = 0, len = layers.length -1; i < len; i++) {
        z = numeric.add(numeric.dot(data , BWts.weights[i]), replicate(BWts.biases[i],Len));
        zs[i] = z;
        data  = sigmoid(z);
        activations[i+1] = data;
    }
    // backward pass

    let delta = numeric.mul(numeric.sub(activations[activations.length -1] ,label),sigmoidDerivative(zs[zs.length-1]));
    temp.biases[BWts.biases.length -1 ] = numeric.dot(numeric.rep([1,Len],1),delta);
    temp.weights[BWts.weights.length -1 ] = numeric.dot(numeric.transpose(activations[activations.length -2 ]), delta) ;
    for ( let layer = 2; layer < layers.length; layer++){
        z = zs[zs.length -layer];
        delta = numeric.mul(numeric.dot(delta , numeric.transpose(BWts.weights[BWts.weights.length -layer + 1])),sigmoidDerivative(z));
        temp.biases[BWts.biases.length -layer ] = numeric.dot(numeric.rep([1,Len],1),delta);
        temp.weights[BWts.biases.length -layer ] =   numeric.dot(numeric.transpose(activations[activations.length -layer -1  ]) , delta);
    }
}
function sgdMiniBatch( data, label){
    let  temp = new zerosMatrixFast();
    backprop(data, label, temp);
    for(let i = 0;i <temp.biases.length;i++){
        BWts.weights[i] =  numeric.sub(BWts.weights[i] ,numeric.mul(eta/ batchSize , temp.weights[i]));
        BWts.biases[i] =  numeric.sub(BWts.biases[i] , numeric.mul(eta/ batchSize, temp.biases[i]));
    }
}
// Call this method once and then comment
//downloadData();
var startTime = Date.now();
var endTime;
readData();
var  BWts = new initializeWeights();
var output, countArr, acc;
for (let epoch =0 ; epoch < epochs ; epoch++){
    [ trainData, trainLbl]= shuffle(trainData, trainLbl);
    for (let j = 0 ;j <trainData.length; j+= batchSize){
        if(j + batchSize <trainData.length){
            sgdMiniBatch(trainData.slice(j, j + batchSize),trainLbl.slice(j, j + batchSize));

        }
        else{

            sgdMiniBatch(trainData.slice(j),trainLbl.slice(j));
        }
    }
    output = feedforward(testData);
    countArr = output.map(function(obj) {
        let indexOfMaxValue = obj.reduce((iMax, x, i, arr) => x > arr[iMax] ? i : iMax, 0);
        return indexOfMaxValue;
    });
    acc  = numeric.sub(countArr,testLbl).filter( tgt => tgt ==0).length *100 / testLbl.length
    endTime = Date.now();
    console.log("Epoch No :", epoch, " Accuracy :",acc,"Time(second)",(endTime - startTime)/1000);
    startTime = endTime;
}
