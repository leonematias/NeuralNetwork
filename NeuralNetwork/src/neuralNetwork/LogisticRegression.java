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
    
    public void train(List<Input> input) {
        float currentAlpha = alpha;
        for (int i = 0; i < iterations; i++) {
            train(input, currentAlpha);
            currentAlpha *= 0.95;
        }
    }
    
    public int predict(float[] x ) {
        float[] predictedY = output(x, tempPrediction);
        return maxIndex(predictedY);
    }
    
    private void train(List<Input> input, float alpha) {
        //Accumulate gradient for all input
        set(weightGrad, 0);
        set(biasGrad, 0);
        int n = input.size();
        for (Input in : input) {
            float[] predictedY = output(in.x, tempPrediction);
            float[] trainY = toBinaryVec(in.y, tempYVec);
            
            //Compute diff: dy = predictedY - trainY
            float[] dy = sub(predictedY, trainY, tempDyVec);
            
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
        set(out, 0);
        
        for (int j = 0; j < outputSize; j++) {
            for (int i = 0; i < inputSize; i++) {
                out[j] += weights[j][i] * x[i];
            }
            out[j] += bias[j];
        }
        
        return softmax(out, outputSize);
    }
    
    private float[] softmax(float[] x, int n) {
        float max = max(x);
        float sum = 0;
        for (int i = 0; i < n; i++) {
            x[i] = (float)Math.exp(x[i] - max);
            sum += x[i];
        }
        div(x, sum);
        return x;
    }
    
    private void set(float[][] m, float s) {
        for (int i = 0; i < m.length; i++) {
            set(m[i], s);
        }
    }
    
    private void set(float[] v, float s) {
        Arrays.fill(v, s);
    }
    
    private int maxIndex(float[] v) {
        float max = Float.NEGATIVE_INFINITY;
        int maxIndex = -1;
        for (int i = 0; i < v.length; i++) {
            if(v[i] > max) {
                max = v[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }
    
    private float max(float[] v) {
        return v[maxIndex(v)];
    }
    
    private float[] mul(float[] v, float s) {
        for (int i = 0; i < v.length; i++) {
            v[i] *= s;
        }
        return v;
    }
    
    private float[] div(float[] v, float s) {
        for (int i = 0; i < v.length; i++) {
            v[i] /= s;
        }
        return v;
    }
    
    private float[] sub(float[] a, float[] b, float[] out) {
        for (int i = 0; i < a.length; i++) {
            out[i] = a[i] - b[i];
        }
        return out;
    }
    
    private float[] toBinaryVec(int i, float[] out) {
        set(out, 0);
        out[i] = 1;
        return out;
    }
    
    public static class Input {
        public final float[] x;
        public final int y;

        public Input(float[] x, int y) {
            this.x = x;
            this.y = y;
        }
    }
    
    
    
    
    public static void main(String[] args) {
        List<Input> trainning = new ArrayList<>(200);
        for (int i = 0; i < 100; i++) {
            trainning.add(new Input(new float[]{-10, -50 + i}, 0));
        }
        for (int i = 0; i < 100; i++) {
            trainning.add(new Input(new float[]{10, -50 + i}, 1));
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
