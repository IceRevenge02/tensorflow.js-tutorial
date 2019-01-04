function onload() {
    const data = createData(2, 1, 6, 255);
    data.print();

    tf.scalar(4).print();//number
    tf.tensor1d([1, 3, 5, 0]).print();//vector
    tf.tensor2d([1, 3, 1, 2], [2, 2]).print();//matrix
    tf.tensor3d([1, 3, 1, 2, 4, 5, 1, 0], [2, 2, 2]).print();//many matrix
}

function createData(nr, rows, cols, max) {
    let values = [];
    for (let i = 0; i < nr * rows * cols; i++) {
        values[i] = Math.round(Math.random()*max);
    }

    const shape = [nr, rows, cols];

    const data = tf.tensor(values, shape, 'int32'); //values:[data], shape: [nr, rows, cols], dtype

    return data;
}