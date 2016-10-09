package neuralNetwork;

import java.util.Arrays;
import java.util.List;

/**
 * Neural Network with one hidden layer.
 * Trainning is done with gradient descent with regularization.
 * 
 * @author Matias Leone
 */
public class NeuralNetwork {

    private final static float RAND_EPSILON = 0.12f;
    
    private final int inputLayerSize;
    private final int internalLayerSize;
    private final int outputLayerSize;
    
    private final float[][] theta0;
    private final float[][] theta1;
    
    //Temp variables
    private final float[] z1;
    private final float[] a1;
    private final float[] z2;
    private final float[] a2;
    private final float[] delta1;
    private final float[] delta2;
    private final float[] y;

    /**
     * Define network topology
     */
    public NeuralNetwork(int inputLayerSize, int internalLayerSize, int outputLayerSize) {
        this.inputLayerSize = inputLayerSize;
        this.internalLayerSize = internalLayerSize;
        this.outputLayerSize = outputLayerSize;
        
        theta0 = new float[internalLayerSize][inputLayerSize + 1];
        theta1 = new float[outputLayerSize][internalLayerSize + 1];
        
        z1 = new float[internalLayerSize];
        a1 = new float[internalLayerSize];
        z2 = new float[outputLayerSize];
        a2 = new float[outputLayerSize];
        delta1 = new float[internalLayerSize + 1];
        delta2 = new float[outputLayerSize];
        y = new float[outputLayerSize];
    }
    
    
    
    /**
     * Train neural network with the given input an expected inputClass.
     * Use gradient descent to minimize cost of weights in theta0 and theta1
     */
    public void train(List<float[]> input, int[] inputClass, int iterations, float alpha, float lambda) {
        if(input.size() != inputClass.length)
            throw new RuntimeException("Invalid input size");
        
        //Init theta0 and and theta1 with random values
        randomInit(theta0, RAND_EPSILON);
        randomInit(theta1, RAND_EPSILON);
        
        //Init theta temp variables
        float[][] theta0Temp = new float[internalLayerSize][inputLayerSize + 1];
        float[][] theta1Temp = new float[outputLayerSize][internalLayerSize + 1];
        set(theta0Temp, 0);
        set(theta1Temp, 0);
        
        //Init gradients
        float[][] theta0Grad = new float[internalLayerSize][inputLayerSize + 1];
        float[][] theta1Grad = new float[outputLayerSize][internalLayerSize + 1];
        set(theta0Grad, 0);
        set(theta1Grad, 0);
        
        //Init cost
        float minCost = 0;
        
        //Perform gradient descent
        for (int n = 0; n < iterations; n++) {
            
            //Compute gradient and cost with current theta
            float cost = computeGradient(input, inputClass, theta0, theta1, lambda, theta0Grad, theta1Grad);
            
            //Update theta0 in temp variable
            for (int i = 0; i < theta0Temp.length; i++) {
                for (int j = 0; j < theta0Temp[i].length; j++) {
                    theta0Temp[i][j] -= alpha * theta0Grad[i][j];
                }
            }
            
            //Update theta1 in temp variable
            for (int i = 0; i < theta1Temp.length; i++) {
                for (int j = 0; j < theta1Temp[i].length; j++) {
                    theta1Temp[i][j] -= alpha * theta1Grad[i][j];
                }
            }
            
            //Set theta0 and theta1
            set(theta0, theta0Temp);
            set(theta1, theta1Temp);
            
            //Update current cost
            minCost = cost;
            
            //System.out.println(n + ") Cost: " + minCost);
        }
    }
    
    /**
     * Compute gradient (theta0Grad, theta1Grad) and cost for the given input
     */
    private float computeGradient(List<float[]> input, int[] inputClass, float[][] theta0, float[][] theta1, float lambda, float[][] theta0Grad, float[][] theta1Grad) {
        //Compute theta0Grad, theta1Grad and cost for inputs
        float cost = 0;
        for (int i = 0; i < input.size(); i++) {
            float[] currentInput = input.get(i);
            int currentInputClass = inputClass[i];
            cost += accumulateGradient(currentInput, currentInputClass, theta0, theta1, theta0Grad, theta1Grad);
        }
        
        //Divide gradient by m
        int m = input.size();
        float oneDivM = 1.0f/m;
        mul(theta0Grad, oneDivM);
        mul(theta1Grad, oneDivM);
        
        //Divde cost by m
        cost = cost * oneDivM;
        
        //Regularized theta0Grad: theta0Grad + (lambda/m) * theta0
        float regularizedCost = 0;
        float lambdaDivM = lambda/m;
        for (int i = 0; i < theta0Grad.length; i++) {
            for (int j = 1; j < theta0Grad[i].length; j++) {
                float v = theta0[i][j];
                theta0Grad[i][j] += lambdaDivM * v;
                regularizedCost += v * v;
            }
        }
        
        //Regularized theta1Grad: theta1Grad + (lambda/m) * theta1
        for (int i = 0; i < theta1Grad.length; i++) {
            for (int j = 1; j < theta1Grad[i].length; j++) {
                float v = theta1[i][j];
                theta1Grad[i][j] += lambdaDivM * v;
                regularizedCost += v * v;
            }
        }
        
        //Add regularization to cost: J = J + lambda/2m * JReg
        cost += (lambda / (2.0f * m)) * regularizedCost;
        
        return cost;
    }
    
    
    
    
    private float accumulateGradient(float[] input, int inputClass, float[][] theta0, float[][] theta1, float[][] theta0Grad, float[][] theta1Grad) {
        if(inputClass < 0 || inputClass >= outputLayerSize)
            throw new RuntimeException("Invalid inputClass: " + inputClass);
        
        //Feedforward pass
        feedForward(input);
        
        //Create y vector using class
        set1InIndex(y, inputClass);
        
        //Compute accumulated cost
        float cost = 0;
        for (int i = 0; i < y.length; i++) {
            cost += -y[i] * Math.log(a2[i]) - (1 - y[i]) * Math.log(1 - a2[i]);
        }
        
        
        //Compute delta2 as: a2 - y
        for (int i = 0; i < delta2.length; i++) {
            delta2[i] = a2[i] - y[i];
        }
        
        //Compute delta1 as: theta1' * delta2 * sigmoidDeriv(z1)
        for (int i = 0; i < theta1.length; i++) {
            for (int j = 1; j < theta1[i].length; j++) {
                delta1[j] = theta1[i][j] * delta2[i] * sigmoidDeriv(z1[i]);
            }
        }
        
        
        //Accumulate theta0Grad: theta0Grad + delta1 * a0'
        for (int i = 0; i < theta0Grad.length; i++) {
            for (int j = 1; j < theta0Grad[i].length; j++) {
                theta0Grad[i][j] += delta1[i + 1] * input[j - 1];
            }
        }
        
        //Accumulate theta1Grad: theta1Grad + delta2 * a1'
        for (int i = 0; i < theta1Grad.length; i++) {
            for (int j = 1; j < theta1Grad[i].length; j++) {
                theta1Grad[i][j] += delta2[i] * a1[j - 1];
            }
        }
        
        return cost;
    }
    
    /**
     * Predit symbol for the given input using trainned network
     */
    public Result predict(float[] input) {
        //Feedforward pass
        feedForward(input);
        
        //Get index with largest output
        int index = maxIndex(a2); 
        
        return new Result(index, a2[index]);
    }
    
    
    private void feedForward(float[] input) {
        if(input.length != inputLayerSize)
            throw new RuntimeException("Invalid input size: " + input.length);
        
        //z1
        for (int i = 0; i < z1.length; i++) {
            z1[i] = theta0[i][0] * 1;
        }
        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < z1.length; j++) {
                z1[j] += theta0[j][i + 1] * input[i];
            }
        }
        
        //a1
        for (int i = 0; i < a1.length; i++) {
            a1[i] = sigmoid(z1[i]);
        }
        
        
        //z2
        for (int i = 0; i < z2.length; i++) {
            z2[i] = theta1[i][0] * 1;
        }
        for (int i = 0; i < a1.length; i++) {
            for (int j = 0; j < z2.length; j++) {
                z2[j] += theta1[j][i + 1] * a1[i];
            }
        }
        
        //a2
        for (int i = 0; i < a2.length; i++) {
            a2[i] = sigmoid(z2[i]);
        }
    }
    
    
    private float sigmoid(float z) {
        return 1 / (1 + (float)Math.exp(-z));
    }
    
    private float sigmoidDeriv(float z) {
        float g = sigmoid(z);
        return g * (1 - g);
    }
    
    private int maxIndex(float[] a) {
        float max = Float.MIN_VALUE;
        int idx = -1;
        for (int i = 0; i < a.length; i++) {
            if(a[i] > max) {
                max = a[i];
                idx = i;
            }
        }
        return idx;
    }
    
    private void set(float[] a, float v) {
        Arrays.fill(a, v);
    }
    
    private void set(float[][] m, float v) {
        for (int i = 0; i < m.length; i++) {
            set(m[i], v);
        }
    }
    
    private void set(float[][] m, float[][] v) {
        for (int i = 0; i < m.length; i++) {
            for (int j = 0; j < m[i].length; j++) {
                m[i][j] = v[i][j];
            }
        }
    }
    
    private void randomInit(float[][] m, float epsilon) {
        float epsilon2 = 2 * epsilon;
        for (int i = 0; i < m.length; i++) {
            for (int j = 0; j < m[i].length; j++) {
                m[i][j] = (float)Math.random() * epsilon2 - epsilon;
            }
        }
    }
    
    private void set1InIndex(float[] y, int index) {
        set(y, 0);
        y[index] = 1;
    }
    
    private void mul(float[][] m, float s) {
        for (int i = 0; i < m.length; i++) {
            mul(m[i], s);
        }
    }
    
    private void mul(float[] a, float s) {
        for (int i = 0; i < a.length; i++) {
            a[i] *= s;
        }
    }
    
    /**
     * Prediction result
     */
    public static class Result {
        public final int predictedClass;
        public final float confidence;

        public Result(int predictedClass, float confidence) {
            this.predictedClass = predictedClass;
            this.confidence = confidence;
        }

    }
    
}
