package neuralNetwork;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Logistic Regression
 * 
 * @author matias.leone
 */
public class LogisticRegression {
    
    private final int inputSize;
    private final int outputSize;
    private final float[][] weights;
    private final float[] bias;
    private final float alpha;
    private final int iterations;
    
    //Temp variables
    private final float[][] weightGrad;
    private final float[] biasGrad;
    private final float[] tempPrediction;
    private final float[] tempYVec;
    private final float[] tempDyVec;
    
    public LogisticRegression(int inputSize, int outputSize, float alpha, int iterations) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.alpha = alpha;
        this.iterations = iterations;
        this.weights = new float[outputSize][inputSize];
        this.bias = new float[outputSize];
        
        this.weightGrad = new float[outputSize][inputSize];
        this.biasGrad = new float[outputSize];
        this.tempPrediction = new float[outputSize];
        this.tempYVec = new float[outputSize];
        this.tempDyVec = new float[outputSize];
    }
    
    public void train(List<MathUtils.Input> input) {
        float currentAlpha = alpha;
        for (int i = 0; i < iterations; i++) {
            train(input, currentAlpha);
            currentAlpha *= 0.95;
        }
    }
    
    public int predict(float[] x) {
        float[] predictedY = output(x, tempPrediction);
        return MathUtils.maxIndex(predictedY);
    }
    
    private void train(List<MathUtils.Input> input, float alpha) {
        //Accumulate gradient for all input
        MathUtils.set(weightGrad, 0);
        MathUtils.set(biasGrad, 0);
        int n = input.size();
        for (MathUtils.Input in : input) {
            float[] predictedY = output(in.x, tempPrediction);
            float[] trainY = MathUtils.toBinaryVec(in.y, tempYVec);
            
            //Compute diff: dy = predictedY - trainY
            float[] dy = MathUtils.sub(predictedY, trainY, tempDyVec);
            
            //Accumulate gradient
            for (int j = 0; j < outputSize; j++) {
                for (int i = 0; i < inputSize; i++) {
                    weightGrad[j][i] += dy[j] * in.x[i];
                }
                biasGrad[j] += dy[j];
            }
        }
        
        //Update weights
        for (int j = 0; j < outputSize; j++) {
            for (int i = 0; i < inputSize; i++) {
                weights[j][i] -= alpha * weightGrad[j][i] / n;
            }
            bias[j] -= alpha * biasGrad[j] / n;
        }
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
    
    
    public static void main(String[] args) {
        List<MathUtils.Input> trainning = new ArrayList<>(200);
        for (int i = 0; i < 100; i++) {
            trainning.add(new MathUtils.Input(new float[]{-10, -50 + i}, 0));
        }
        for (int i = 0; i < 100; i++) {
            trainning.add(new MathUtils.Input(new float[]{10, -50 + i}, 1));
        }
        
        LogisticRegression classifier = new LogisticRegression(2, 2, 0.2f, 2000);
        classifier.train(trainning);
        
        
        System.out.println("Expected class: 0");
        System.out.println(classifier.predict(new float[]{-10, 0}));
        System.out.println(classifier.predict(new float[]{-10, 50}));
        System.out.println(classifier.predict(new float[]{-10, -50}));
        System.out.println(classifier.predict(new float[]{-10, 20}));
        System.out.println(classifier.predict(new float[]{-10, -20}));
        System.out.println(classifier.predict(new float[]{-10, 100}));
        System.out.println(classifier.predict(new float[]{-10, -100}));
        System.out.println(classifier.predict(new float[]{-10, 500}));
        System.out.println(classifier.predict(new float[]{-5, 0}));
        System.out.println(classifier.predict(new float[]{-2, 0}));
        System.out.println(classifier.predict(new float[]{-1, 0}));
        
        System.out.println("Expected class: 1");
        System.out.println(classifier.predict(new float[]{10, 0}));
        System.out.println(classifier.predict(new float[]{10, 50}));
        System.out.println(classifier.predict(new float[]{10, -50}));
        System.out.println(classifier.predict(new float[]{10, 20}));
        System.out.println(classifier.predict(new float[]{10, -20}));
        System.out.println(classifier.predict(new float[]{10, 100}));
        System.out.println(classifier.predict(new float[]{10, -100}));
        System.out.println(classifier.predict(new float[]{10, 500}));
        System.out.println(classifier.predict(new float[]{5, 0}));
        System.out.println(classifier.predict(new float[]{2, 0}));
        System.out.println(classifier.predict(new float[]{1, 0}));
    }
    
    
}
