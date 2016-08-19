/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralNetwork;

/**
 *
 * @author Matias Leone
 */
public class NeuralNetwork {

    private final static int INTERNAL_LAYER_COUNT = 25;
    private final static int OUTPUT_LAYER_COUNT = 10;
    
    private int inputLength;
    private float[][] theta0;
    private float[][] theta1;
    
    /*
        Theta0 has size 25 x 401
        Theta1 has size 10 x 26
    */
    
    public Result predict(float[] input) {
        
        
        //z1
        float[] z1 = new float[INTERNAL_LAYER_COUNT];
        for (int i = 0; i < z1.length; i++) {
            z1[i] = theta0[i][0] * 1;
        }
        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < z1.length; j++) {
                z1[j] += theta0[j][i + 1] * input[i];
            }
        }
        
        //a1
        float[] a1 = new float[INTERNAL_LAYER_COUNT];
        for (int i = 0; i < a1.length; i++) {
            a1[i] = sigmoid(z1[i]);
        }
        
        
        //z2
        float[] z2 = new float[OUTPUT_LAYER_COUNT];
        for (int i = 0; i < z2.length; i++) {
            z2[i] = theta1[i][0] * 1;
        }
        for (int i = 0; i < a1.length; i++) {
            for (int j = 0; j < z2.length; j++) {
                z2[j] += theta1[j][i + 1] * a1[i];
            }
        }
        
        //a2
        float[] a2 = new float[OUTPUT_LAYER_COUNT];
        for (int i = 0; i < a2.length; i++) {
            a2[i] = sigmoid(z2[i]);
        }
        
        
        //Get index with largest output
        int index = maxIndex(a2);
        
        
        return new Result(index, a2[index]);
    }
    
    
    private float sigmoid(float z) {
        return 1 / (1 + (float)Math.exp(-z));
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
    
    public static class Result {
        public final int predictedClass;
        public final float confidence;

        public Result(int predictedClass, float confidence) {
            this.predictedClass = predictedClass;
            this.confidence = confidence;
        }
        
        
    }
    
}
