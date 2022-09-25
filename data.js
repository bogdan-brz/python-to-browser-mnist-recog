const IMAGE_SIZE = 784;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;

const TRAIN_TEST_RATIO = 5 / 6;

const NUM_TRAIN_ELEMENTS = Math.floor(TRAIN_TEST_RATIO * NUM_DATASET_ELEMENTS);
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

const MNIST_IMAGES_SPRITE_PATH =
    "https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png";
const MNIST_LABELS_PATH =
    "https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8";

export class MnistData {
    constructor() {
        this.shuffledTrainIndex = 0;
        this.shuffledTestIndex = 0;
    }

    async load() {
        // Make a request for the MNIST sprited image.
        const img = new Image();
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");
        const imgRequest = new Promise((resolve, reject) => {
            img.crossOrigin = ""; //allows "cors", so the canvas doesn't get tainted when an outside image is set on it
            img.onload = () => {
                // this happens after the img src is set
                img.width = img.naturalWidth; // allocating space for image - just a good thing to do, doesn't have any functionality
                img.height = img.naturalHeight;

                const datasetBytesBuffer = new ArrayBuffer(
                    NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4
                ); // creates an array buffer, which is a space of data, it can only be written to using a "view" such as a float32array (datasetBytesView).
                // The * 4 is there because it's length is in bytes and you need 4 bytes for a 32float which is what this buffer is used for

                const chunkSize = 5000;
                canvas.width = img.width;
                canvas.height = chunkSize;

                for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
                    // divides the loading of bytes into chunks to not clog up the browser
                    const datasetBytesView = new Float32Array(
                        datasetBytesBuffer,
                        i * IMAGE_SIZE * chunkSize * 4,
                        IMAGE_SIZE * chunkSize
                    ); // creates a 32float array view which writes to the datasetbytesbuffer.
                    // The offset is current chunk times chunk size times image size times 4 bytes per 32float
                    ctx.drawImage(
                        img,
                        0,
                        i * chunkSize,
                        img.width,
                        chunkSize,
                        0,
                        0,
                        img.width,
                        chunkSize
                    ); // draws a slice (chunksize wide) of the sprite image into the canvas, because it's easier to read pixel data from a canvas

                    const imageData = ctx.getImageData(
                        0,
                        0,
                        canvas.width,
                        canvas.height
                    ); // gets the pixel values into an array (the array is chunksize * image width long)

                    for (let j = 0; j < imageData.data.length / 4; j++) {
                        // it's a grey scale image, and every 4th value is the one we need, so we extract that and normalize it ( / 255)
                        datasetBytesView[j] = imageData.data[j * 4] / 255;
                    }
                }
                this.datasetImages = new Float32Array(datasetBytesBuffer);
                // after putting all the data into the buffer via the chunk-size views, we make a float32 array out of it.
                // The array now is a list of values between 0 - 1 that is "img width (784) times # of images (65000)" long (65000)

                resolve();
            };
            img.src = MNIST_IMAGES_SPRITE_PATH; // this triggers img.onload
        });

        const labelsRequest = fetch(MNIST_LABELS_PATH);
        const [imgResponse, labelsResponse] = await Promise.all([
            imgRequest,
            labelsRequest,
        ]);

        this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());
        // the labels are received and placed into a bufferarray.
        // (the array is 65000 long this time, because you only need one byte to encode a label: 0-9)
        // a uint8 array is created from the buffer, it's 65000 long, and contains integers (only 0s and 1s) (one-hot encoded)

        // Create shuffled indices into the train/test set for when we select a
        // random dataset element for training / validation.
        this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS);
        this.testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS);

        // Slice the the images and labels into train and test sets.
        this.trainImages = this.datasetImages.slice(
            0,
            IMAGE_SIZE * NUM_TRAIN_ELEMENTS
        );
        this.testImages = this.datasetImages.slice(
            IMAGE_SIZE * NUM_TRAIN_ELEMENTS
        );
        this.trainLabels = this.datasetLabels.slice(
            0,
            NUM_CLASSES * NUM_TRAIN_ELEMENTS
        );
        this.testLabels = this.datasetLabels.slice(
            NUM_CLASSES * NUM_TRAIN_ELEMENTS
        );
    }

    nextTrainBatch(batchSize) {
        return this.nextBatch(
            // call the nextBatch function with some arguments: batchsize, data, function to call to get the next train index from the shuffled indeces array
            batchSize,
            [this.trainImages, this.trainLabels],
            () => {
                this.shuffledTrainIndex =
                    (this.shuffledTrainIndex + 1) % this.trainIndices.length; // modulo division just puts the index back at the beginning once it exceeds the array length
                return this.trainIndices[this.shuffledTrainIndex];
            }
        );
    }

    nextTestBatch(batchSize) {
        return this.nextBatch(
            // call the nextBatch function with some arguments: batchsize, data, function to call to get the next test index from the shuffled indeces array
            batchSize,
            [this.testImages, this.testLabels],
            () => {
                this.shuffledTestIndex =
                    (this.shuffledTestIndex + 1) % this.testIndices.length; // modulo division just puts the index back at the beginning once it exceeds the array length
                return this.testIndices[this.shuffledTestIndex];
            }
        );
    }

    nextBatch(batchSize, data, index) {
        const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE);
        const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES);
        // respective arrays to hold the batch's images and labels

        for (let i = 0; i < batchSize; i++) {
            const idx = index(); // gets the new index from the shuffled indeces array

            const image = data[0].slice(
                idx * IMAGE_SIZE,
                idx * IMAGE_SIZE + IMAGE_SIZE
            ); // getting a slice of the images array (could be train or test) that's imagesize long from the shuffled index spot,
            // and putting it on the next cossecutive spot in the newly created batch images array
            batchImagesArray.set(image, i * IMAGE_SIZE);

            const label = data[1].slice(
                idx * NUM_CLASSES,
                idx * NUM_CLASSES + NUM_CLASSES
            ); // extracts the label at index spot and places it in the new batch labels array. it's times num_classes because lables are one-hot encoded
            batchLabelsArray.set(label, i * NUM_CLASSES);
        }

        const xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE]);
        const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES]); // creates tensors from the batch arrays

        return { xs, labels };
    }
}
