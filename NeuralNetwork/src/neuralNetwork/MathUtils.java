package neuralNetwork;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 *
 * @author Matias Leone
 */
public class MathUtils {
    
    public static final Random RAND = new Random();
    
    public static class Input {
        public final float[] x;
        public int y;

        public Input(float[] x, int y) {
            this.x = x;
            this.y = y;
        }
    }
    
    
    public static float[] softmax(float[] x, int n) {
        float max = max(x);
        float sum = 0;
        for (int i = 0; i < n; i++) {
            x[i] = (float)Math.exp(x[i] - max);
            sum += x[i];
        }
        div(x, sum);
        return x;
    }
    
    public static float sigmoid(float x) {
        return 1f / (1f + (float)Math.pow(Math.E, -x));
    }
    
    public static float[] sigmoid(float[] x) {
        for (int i = 0; i < x.length; i++) {
            x[i] = sigmoid(x[i]);
        }
        return x;
    }
    
    public static float dsigmoid(float x) {
        return x * (1f - x);
    }
    
    public static float[] dsigmoid(float[] x) {
        for (int i = 0; i < x.length; i++) {
            x[i] = dsigmoid(x[i]);
        }
        return x;
    }
    
    public static float[][] set(float[][] m, float s) {
        for (int i = 0; i < m.length; i++) {
            set(m[i], s);
        }
        return m;
    }
    
    public static float[] set(float[] v, float s) {
        Arrays.fill(v, s);
        return v;
    }
    
    public static int maxIndex(float[] v) {
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
    
    public static float max(float[] v) {
        return v[maxIndex(v)];
    }
    
    public static float[] mul(float[] v, float s) {
        for (int i = 0; i < v.length; i++) {
            v[i] *= s;
        }
        return v;
    }
    
    public static float[] div(float[] v, float s) {
        for (int i = 0; i < v.length; i++) {
            v[i] /= s;
        }
        return v;
    }
    
    public static float[] sub(float[] a, float[] b, float[] out) {
        for (int i = 0; i < a.length; i++) {
            out[i] = a[i] - b[i];
        }
        return out;
    }
    
    public static float[] toBinaryVec(int i, float[] out) {
        set(out, 0);
        out[i] = 1;
        return out;
    }
    
    public static void setRandomValues(float[][] mat, float min, float max) {
        float diff = max - min;
        for (int i = 0; i < mat.length; i++) {
            for (int j = 0; j < mat[i].length; j++) {
                mat[i][j] = RAND.nextFloat() * diff + min;
            }
        }
    }
    
    public static List<MathUtils.Input> createEmptyInput(int itemsCount, int featureSize) {
        List<MathUtils.Input> input = new ArrayList<>(itemsCount);
        for (int i = 0; i < itemsCount; i++) {
            input.add(new Input(new float[featureSize], 0));
        }
        return input;
    }
    
    public static List<MathUtils.Input> clone(List<MathUtils.Input> orig) {
        List<MathUtils.Input> clone = new ArrayList<>(orig.size());
        for (int i = 0; i < orig.size(); i++) {
            MathUtils.Input item = orig.get(i);
            clone.add(new Input(Arrays.copyOf(item.x, item.x.length), item.y));
        }
        return clone;
    }
    
    public static void setFeatures(List<MathUtils.Input> list, float featureValue) {
        for (Input item : list) {
            set(item.x, featureValue);
        }
    }
    
    
}
