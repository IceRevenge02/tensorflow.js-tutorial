function setup() {
    noCanvas();
    frameRate(5);
}

function draw() {
    tf.tidy(() => { //stops memory leak
        const a = createTensor(1, 200, 600, 255); //Values can never be changed
        const b = createTensor(1, 600, 200, 255); //Values can never be changed

        const c = a.matMul(b);
    }); 
    const t = tf.scalar(1); //this one is not getting tidied, so numTensors rises per tick
    console.log(tf.memory().numTensors);
}

function createTensor(nr, rows, cols, max) {
    let values = [];
    for (let i = 0; i < nr * rows * cols; i++) {
        values[i] = Math.round(Math.random()*max);
    }

    const shape = [nr, rows, cols];

    const t = tf.tensor(values, shape, 'int32'); //values:[data], shape: [nr, rows, cols], dtype

    return t;
}