package neuralNetwork;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 *
 * @author Matias Leone
 */
public class NeuralNetwork2 {

    private final int inputLayerSize;
    private final int internalLayerSize;
    private final int outputLayerSize;
    private final float alpha;
    private final int iterations;
    private final HiddenLayer hiddenLayer;
    private final LogisticRegression logisticLayer;
    
    //Temp variables
    private final float[] tempPrediction1;
    private final float[] tempPrediction2;
    
    public NeuralNetwork2(int inputLayerSize, int internalLayerSize, int outputLayerSize, float alpha, int iterations) {
        this.inputLayerSize = inputLayerSize;
        this.internalLayerSize = internalLayerSize;
        this.outputLayerSize = outputLayerSize;
        this.alpha = alpha;
        this.iterations = iterations;
        
        this.hiddenLayer = new HiddenLayer(inputLayerSize, internalLayerSize);
        this.logisticLayer = new LogisticRegression(internalLayerSize, outputLayerSize);
        
        this.tempPrediction1 = new float[internalLayerSize];
        this.tempPrediction2 = new float[outputLayerSize];
    }
    
    public void train(List<MathUtils.Input> input) {
        //Create reusable temp variables for all iterations
        int n = input.size();
        float[][] dyVec = new float[n][outputLayerSize];
        List<MathUtils.Input> z = new ArrayList<>(n);
        for (MathUtils.Input in : input) {
            z.add(new MathUtils.Input(new float[internalLayerSize], in.y));
        }
        
        //Train N iterations
        for (int i = 0; i < iterations; i++) {
            train(input, z, dyVec);
        }
    }
    
    public int predict(float[] x) {
        float[] z = hiddenLayer.output(x, tempPrediction1);
        float[] predictedY = logisticLayer.output(z, tempPrediction2);
        return MathUtils.maxIndex(predictedY);
    }
    
    private void train(List<MathUtils.Input> input, List<MathUtils.Input> z, float[][] dyVec) {
        MathUtils.setFeatures(z, 0);
        
        //Compute hidden layer output (z) for all items
        for (int i = 0; i < input.size(); i++) {
            MathUtils.Input in = input.get(i);
            
            //Forward hidden layer
            hiddenLayer.output(in.x, z.get(i).x);
        }
        
        //Train output layer with logistic regression
        float[][] dy = logisticLayer.train(z, alpha, dyVec);
        
        //Backward propagation to hidden layer
        hiddenLayer.backward(input, z, dy, logisticLayer.weights, alpha);
    }
    
    
    
    
    
    private class HiddenLayer {
        private final int inputSize;
        private final int outputSize;
        private final float[][] weights;
        private final float[] bias;
        
        //Temp vatiables
        private final float[][] weightGrad;
        private final float[] biasGrad;
        private final float[] tempDzVec;

        public HiddenLayer(int inputSize, int outputSize) {
            this.inputSize = inputSize;
            this.outputSize = outputSize;
            this.weights = new float[outputSize][inputSize];
            this.bias = new float[outputSize];
            
            //Init weights
            float w = 1f / inputSize;
            MathUtils.setRandomValues(this.weights, -w, w);
            
            this.weightGrad = new float[outputSize][inputSize];
            this.biasGrad = new float[outputSize];
            this.tempDzVec = new float[outputSize];
        }
        
        public float[] output(float[] x, float[] out) {
            MathUtils.set(out, 0);

            for (int j = 0; j < outputSize; j++) {
                for (int i = 0; i < inputSize; i++) {
                    out[j] += weights[j][i] * x[i];
                }
                out[j] += bias[j];
            }

            return MathUtils.sigmoid(out);
        }
        
        public void backward(List<MathUtils.Input> input, List<MathUtils.Input> z, float[][] dy, float[][] weightsPrev, float alpha) {
            MathUtils.set(weightGrad, 0);
            MathUtils.set(biasGrad, 0);
            
            //Accumulate gradient for all input
            int n = input.size();
            int prevOutLayerSize = dy[0].length;
            for (int m = 0; m < n; m++) {
                MathUtils.Input in = input.get(m);
                float[] dz = MathUtils.set(tempDzVec, 0);
                
                //Propagate error backward
                for (int j = 0; j < outputSize; j++) {
                    
                    //Backpropagation
                    for (int k = 0; k < prevOutLayerSize; k++) {
                        dz[j] += weightsPrev[k][j] * dy[m][k];
                    }
                    dz[j] *= MathUtils.dsigmoid(z.get(m).x[j]);
   
                    //Acumulate gradient
                    for (int i = 0; i < inputSize; i++) {
                        weightGrad[j][i] += dz[j] * in.x[i];
                    }
                    biasGrad[j] += dz[j];  
                }
            }
            
            //Update weights with gradient descent
            for (int j = 0; j < outputSize; j++) {
                for (int i = 0; i < inputSize; i++) {
                    weights[j][i] -= alpha * weightGrad[j][i] / n;
                }
                bias[j] -= alpha * biasGrad[j] / n;
            }
            
        }
        
    }
    
    
    private class LogisticRegression {
        private final int inputSize;
        private final int outputSize;
        private final float[][] weights;
        private final float[] bias;
        
        //Temp vatiables
        private final float[][] weightGrad;
        private final float[] biasGrad;
        private final float[] tempPrediction;
        private final float[] tempYVec;
        
        public LogisticRegression(int inputSize, int outputSize) {
            this.inputSize = inputSize;
            this.outputSize = outputSize;
            this.weights = new float[outputSize][inputSize];
            this.bias = new float[outputSize];
            
            this.weightGrad = new float[outputSize][inputSize];
            this.biasGrad = new float[outputSize];
            this.tempPrediction = new float[outputSize];
            this.tempYVec = new float[outputSize];
        }
        
        public float[][] train(List<MathUtils.Input> input, float alpha, float[][] dyVec) {
            //Accumulate gradient for all input
            MathUtils.set(weightGrad, 0);
            MathUtils.set(biasGrad, 0);
            MathUtils.set(dyVec, 0);
            int n = input.size();
            for (int k = 0; k < n; k++) {
                MathUtils.Input in = input.get(k);
                
                //Predict value
                float[] predictedY = output(in.x, tempPrediction);
                float[] trainY = MathUtils.toBinaryVec(in.y, tempYVec);

                //Compute diff: dy = predictedY - trainY
                float[] dy = MathUtils.sub(predictedY, trainY, dyVec[k]);

                //Accumulate gradient
                for (int j = 0; j < outputSize; j++) {
                    for (int i = 0; i < inputSize; i++) {
                        weightGrad[j][i] += dy[j] * in.x[i];
                    }
                    biasGrad[j] += dy[j];
                }
            }

            //Update weights with gradient descent
            for (int j = 0; j < outputSize; j++) {
                for (int i = 0; i < inputSize; i++) {
                    weights[j][i] -= alpha * weightGrad[j][i] / n;
                }
                bias[j] -= alpha * biasGrad[j] / n;
            }
            
            //Return diff vector for all inputs
            return dyVec;
        }
        
        private float[] output(float[] x, float[] out) {
            MathUtils.set(out, 0);

            for (int j = 0; j < outputSize; j++) {
                for (int i = 0; i < inputSize; i++) {
                    out[j] += weights[j][i] * x[i];
                }
                out[j] += bias[j];
            }

            return MathUtils.softmax(out, outputSize);
        }

    }
    
    
    
    
    
    
}
