/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralNetwork;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.awt.event.MouseMotionListener;
import java.awt.image.BufferedImage;
import javax.swing.JPanel;

/**
 *
 * @author matias.leone
 */
public class DrawingPanel extends JPanel {
    
    private final static int POINT_RAD = 10;
    
    private BufferedImage renderImg;
    private Graphics2D renderG;
    private Dimension graphDim;
    private boolean drawing;
    
    public DrawingPanel() {
        this.drawing = false;
        
        this.addMouseMotionListener(new MouseMotionAdapter() {
            @Override
            public void mouseDragged(MouseEvent e) {
                if(drawing) {
                    drawPoint(e.getX(), e.getY());
                }
            }
        });
        this.addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                drawing = true;
            }
            
            @Override
            public void mouseExited(MouseEvent e) {
                drawing = false;
            }

            @Override
            public void mouseClicked(MouseEvent e) {
                drawPoint(e.getX(), e.getY());
            }
            
            
        });
        
        
    }
    
    @Override
    public void paint(Graphics g){
            update(g);
    }

    @Override
    public void update(Graphics g) {

        if(renderImg == null) {
            graphDim = getSize();
            renderImg = (BufferedImage)createImage(graphDim.width, graphDim.height);
            renderG = renderImg.createGraphics();
            
            renderG.setPaint(Color.WHITE);
            renderG.fillRect(0, 0, graphDim.width, graphDim.height);
        }

        g.drawImage(renderImg, 0, 0, this);
    } 

    
    private void drawPoint(int x, int y) {
        renderG.setPaint(Color.BLACK);
        renderG.fillOval(x - POINT_RAD, y - POINT_RAD, POINT_RAD * 2, POINT_RAD * 2);
        this.repaint();
    }
    
    public void clearDraw() {
        renderG.setPaint(Color.WHITE);
        renderG.fillRect(0, 0, graphDim.width, graphDim.height);
        this.repaint();
    }
    
    public BufferedImage createImage() {
        BufferedImage image = new BufferedImage(graphDim.width, graphDim.height, BufferedImage.TYPE_INT_ARGB);
        Graphics imageG = image.createGraphics();
        imageG.drawImage(renderImg, 0, 0, this);  
        return image;
    }
}
