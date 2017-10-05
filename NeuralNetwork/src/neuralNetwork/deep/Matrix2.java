package neuralNetwork.deep;

import java.util.Arrays;

/**
 * A nxm float immutable matrix.
 * Operations are not optimized (for acadamedic purposes only)
 * 
 * @author Matias Leone
 */
public class Matrix2 {
    
    private final float[] data;
    private final int rows;
    private final int cols;
    
    public Matrix2(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new float[rows * cols];
    }
    
    public static Matrix2 fromValue(int rows, int cols, float v) {
        Matrix2 m = new Matrix2(rows, cols);
        Arrays.fill(m.data, v);
        return m;
    }
    
    public static Matrix2 zeros(int rows, int cols) {
        return fromValue(rows, cols, 0);
    }
    
    public static Matrix2 random(int rows, int cols) {
        return new Matrix2(rows, cols).apply(RandomOp.INSTANCE);
    }
    
    private int pos(int row, int col) {
        return row * this.cols + col;
    }
    
    private int rowStart(int row) {
        return row * this.cols;
    }
    
    private int rowEnd(int row) {
        return rowStart(row) + this.cols;
    }
    
    private Matrix2 emptyCopy() {
        return new Matrix2(this.rows, this.cols);
    }
    
    public float get(int row, int col) {
        return this.data[pos(row, col)];
    }
    
    public int rows() {
        return this.rows;
    }
    
    public int cols() {
        return this.cols;
    }
    
    public Matrix2 apply(ElementWiseOp op) {
        Matrix2 m = this.emptyCopy();
        for (int row = 0; row < this.rows; row++) {
            for (int col = 0; col < this.cols; col++) {
                int pos = pos(row, col);
                m.data[pos] = op.apply(m.data[pos]);
            }
        }
        return m;
    }
    
    public Matrix2 mul(float s) {
        return apply(new MulOp(s));
    }
    
    public Matrix2 add(float s) {
        return apply(new AddOp(s));
    }
    
    public Matrix2 sub(float s) {
        return apply(new SubOp(s));
    }
    
    public Matrix2 div(float s) {
        return apply(new DivOp(s));
    }
    
    public Matrix2 scalarMinus(float s) {
        return apply(new ScalarMinusOp(s));
    }
    
    public Matrix2 oneMinus() {
        return apply(ScalarMinusOp.ONE_MINUS);
    }
    
    public Matrix2 log() {
        return apply(LogOp.INSTANCE);
    }
    
    public Matrix2 sigmoid() {
        return apply(SigmoidOp.INSTANCE);
    }
    
    public Matrix2 relu() {
        return apply(ReluOp.INSTANCE);
    }
    
    public Matrix2 mul(Matrix2 m) {
        return Matrix2.mul(this, m);
    }
    
    public Matrix2 add(Matrix2 m) {
        return Matrix2.add(this, m);
    }
    
    public Matrix2 sub(Matrix2 m) {
        return Matrix2.sub(this, m);
    }
    
    public Matrix2 mulEW(Matrix2 m) {
        return Matrix2.mulEW(this, m);
    }
    
    public Matrix2 divEW(Matrix2 m) {
        return Matrix2.divEW(this, m);
    }

    @Override
    public String toString() {
        return "(" + rows + ", " + cols + "): " + Arrays.toString(this.data);
    }
    
    public Matrix2 broadcastCol(int cols) {
        if(this.cols == 1)
            throw new RuntimeException("Broadcast not supported for more than 1 column");
        
        Matrix2 m = new Matrix2(this.rows, cols);
        for (int col = 0; col < cols; col++) {
            for (int row = 0; row < this.rows; row++) {
                m.data[pos(row, col)] = this.data[pos(row, 0)];
            }
        }
        return m;
    }
    
    public Matrix2 broadcastRow(int rows) {
        if(this.rows == 1)
            throw new RuntimeException("Broadcast not supported for more than 1 row");
        
        Matrix2 m = new Matrix2(rows, this.cols);
        for (int col = 0; col < this.cols; col++) {
            for (int row = 0; row < rows; row++) {
                m.data[pos(row, col)] = this.data[pos(0, col)];
            }
        }
        return m;
    }
    
    public Matrix2 sumColumns() {
        Matrix2 m = new Matrix2(this.rows, 1);
        for (int row = 0; row < this.rows; row++) {
            int sum = 0;
            for (int col = 0; col < this.cols; col++) {
                sum += this.data[this.pos(row, col)];
            }
            m.data[m.pos(row, 0)] = sum;
        }
        return m;
    }
    
    public Matrix2 transpose() {
        return Matrix2.transpose(this);
    }
    
    
    
    public static Matrix2 mul(Matrix2 a, Matrix2 b) {
        if(a.cols != b.rows)
            throw new RuntimeException("Invalid shapes, a: " + a + ", b: " + b);
        
        Matrix2 c = new Matrix2(a.cols, b.rows);
        for (int row = 0; row < a.rows; row++) {
            for (int col = 0; col < b.cols; col++) {
                c.data[c.pos(row, col)] = rowColumnDot(a, row, b, col);
            }
        }
        return c;
    }
    
    private static float rowColumnDot(Matrix2 a, int row, Matrix2 b, int col) {
        float dot = 0;
        for (int i = 0; i < a.cols; i++) {
            for (int j = 0; j < b.rows; j++) {
                dot += a.get(row, i) * b.get(j, col);
            }
        }
        return dot;
    }
    
    public static Matrix2 add(Matrix2 a, Matrix2 b) {
        if(!sameShape(a, b))
            throw new RuntimeException("Invalid shapes, a: " + a + ", b: " + b);
        
        Matrix2 c = a.emptyCopy();
        for (int row = 0; row < a.rows; row++) {
            for (int col = 0; col < a.cols; col++) {
                int pos = a.pos(row, col);
                c.data[pos] = a.data[pos] + b.data[pos];
            }
        }
        return c;
    }
    
    public static Matrix2 sub(Matrix2 a, Matrix2 b) {
        if(!sameShape(a, b))
            throw new RuntimeException("Invalid shapes, a: " + a + ", b: " + b);
        
        Matrix2 c = a.emptyCopy();
        for (int row = 0; row < a.rows; row++) {
            for (int col = 0; col < a.cols; col++) {
                int pos = a.pos(row, col);
                c.data[pos] = a.data[pos] - b.data[pos];
            }
        }
        return c;
    }
    
    public static boolean sameShape(Matrix2 a, Matrix2 b) {
        return a.rows == b.rows && a.cols == b.cols;
    }
    
    public static Matrix2 mulEW(Matrix2 a, Matrix2 b) {
        if(a.cols != b.rows)
            throw new RuntimeException("Invalid shapes, a: " + a + ", b: " + b);
        
        Matrix2 c = a.emptyCopy();
        for (int row = 0; row < a.rows; row++) {
            for (int col = 0; col < a.cols; col++) {
                int pos = a.pos(row, col);
                c.data[pos] = a.data[pos] * b.data[pos];
            }
        }
        return c;
    }
    
    public static Matrix2 divEW(Matrix2 a, Matrix2 b) {
        if(a.cols != b.rows)
            throw new RuntimeException("Invalid shapes, a: " + a + ", b: " + b);
        
        Matrix2 c = a.emptyCopy();
        for (int row = 0; row < a.rows; row++) {
            for (int col = 0; col < a.cols; col++) {
                int pos = a.pos(row, col);
                c.data[pos] = a.data[pos] / b.data[pos];
            }
        }
        return c;
    }
    
    public static Matrix2 transpose(Matrix2 m) {
        Matrix2 t = m.emptyCopy();
        for (int row = 0; row < m.rows; row++) {
            for (int col = 0; col < m.cols; col++) {
                t.data[t.pos(col, row)] = m.get(row, col);
            }
        }
        return t;
    }

    
    
    /**
     * Element wise operation
     */
    public interface ElementWiseOp {
        float apply(float v);
    }
    
    public static class RandomOp implements ElementWiseOp {
        public static final RandomOp INSTANCE = new RandomOp();
        @Override
        public float apply(float v) {
            return v * (float)Math.random();
        }
    }
    
    public static abstract class ScalarOp implements ElementWiseOp {
        protected final float s;
        public ScalarOp(float s) {
            this.s = s;
        }
    }
    
    public static class MulOp extends ScalarOp {
        public MulOp(float s) {
            super(s);
        }
        @Override
        public float apply(float v) {
            return v * s;
        }
    }
    
    public static class AddOp extends ScalarOp {
        public AddOp(float s) {
            super(s);
        }
        @Override
        public float apply(float v) {
            return v + s;
        }
    }
    
    public static class SubOp extends ScalarOp {
        public SubOp(float s) {
            super(s);
        }
        @Override
        public float apply(float v) {
            return v - s;
        }
    }
    
    public static class DivOp extends ScalarOp {
        public DivOp(float s) {
            super(s);
        }
        @Override
        public float apply(float v) {
            return v / s;
        }
    }
    
    public static class ScalarMinusOp extends ScalarOp {
        public static final ScalarMinusOp ONE_MINUS = new ScalarMinusOp(1);
        public ScalarMinusOp(float s) {
            super(s);
        }
        @Override
        public float apply(float v) {
            return s - v;
        }
    }
    
    public static class SigmoidOp implements ElementWiseOp {
        public static final SigmoidOp INSTANCE = new SigmoidOp();
        @Override
        public float apply(float v) {
            return 1 / (1 + (float)Math.exp(-v));
        }
    }
    
    public static class ReluOp implements ElementWiseOp {
        public static final ReluOp INSTANCE = new ReluOp();
        @Override
        public float apply(float v) {
            return Math.max(0, v);
        }
    }
    
    public static class LogOp implements ElementWiseOp {
        public static final LogOp INSTANCE = new LogOp();
        @Override
        public float apply(float v) {
            return (float)Math.log(v);
        }
    }
    
}
