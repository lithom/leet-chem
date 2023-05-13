package tech.molecules.leet.gui;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.util.ArrayList;
import java.util.List;

public class JSlimRangeSlider extends JPanel {

    public enum EventMode {DRAG_EVENTS,RELEASE_EVENTS}

    private EventMode eventMode = EventMode.RELEASE_EVENTS;

    private double a,b;
    private double rangeA, rangeB;

    private boolean draggingHandle = false;

    /**
     * -1 : no dragged handle, 0 dragging left, 1 dragging right
     */
    private int draggedHandle      = -1;
    /**
     * -1 : mouse not over handle, 0 over left handle, 1 over right handle
     */
    private int mouseOverHandle    = -1;


    int sizeHandle = 10;
    Paint paintSelectedInterval = new Color(20,200,20);
    Paint  paintNonselectedInterval = new Color(40,40,40);

    //Stroke strokeSelectedInterval = new BasicStroke(4);
    Stroke strokeNonselectedInterval = new BasicStroke( 1);

    Paint paintHandleMouseOver = new Color(224,60,20);
    Paint paintHandleMouseNotOver = new Color(20,240,40);


    public JSlimRangeSlider(double a, double b) {
        this.setOpaque(false);
        this.setPreferredSize(new Dimension(120,30));

        this.setDomain(a,b);
        this.setRange(a,b);

        SlimRangeSliderMouseAndMouseMotionListener li = new SlimRangeSliderMouseAndMouseMotionListener();
        this.addMouseListener(li);
        this.addMouseMotionListener(li);
    }

    public void setDomain(double a, double b) {
        this.a = a; this.b = b;
        this.setRange(this.rangeA, this.rangeB);
    }

    public void setRange(double ra, double rb) {
        if(Double.isNaN(ra)) {ra = this.a;}
        if(Double.isNaN(rb)) {rb = this.b;}
        this.rangeA = Math.min( this.b, Math.max( ra , this.a ) );
        this.rangeB = Math.min( this.b, Math.max( rb , this.a ) );
    }


    public double transformXToRangePos(double x) {
        return this.a + (this.b-this.a) *  ((x-0.5*sizeHandle) / (getWidth()-sizeHandle));
    }
    public double transformRangePosToX(double rangepos) {
        return 0.5*sizeHandle + ( (rangepos-this.a) / (this.b-this.a) ) * (getWidth()-sizeHandle);
    }


    public double[] getRange() {
        return new double[]{this.rangeA,this.rangeB};
    }

    private List<ChangeListener> listeners = new ArrayList<>();

    public void addChangeListener(ChangeListener cl) {
        this.listeners.add(cl);
    }

    public void removeChangeListener(ChangeListener cl) {
        this.listeners.remove(cl);
    }

    private void fireChangeEvent() {
        for(ChangeListener cli : this.listeners) {
            cli.stateChanged(new ChangeEvent(this));
        }
    }

    /**
     *
     * @return -1 : not over handle; 0: over left handle, 1: over right handle
     */
    private int isMouseOverHandle(Point p) {
        if( getHandleA().contains(p) ) {
            return 0;
        }
        if( getHandleB().contains(p) ) {
            return 1;
        }
        return -1;
    }

    private Rectangle getHandleA() {
        double interv = this.b-this.a;
        double xa = transformRangePosToX(this.rangeA);//(this.rangeA - this.a) * (this.getWidth() / interv);
        double ya = this.getHeight()/2;
        return new Rectangle( (int) ( xa - 0.5*this.sizeHandle ) , (int) (ya - 0.5*this.sizeHandle) , this.sizeHandle , this.sizeHandle );
    }
    private Rectangle getHandleB() {
        double interv = this.b-this.a;
        double xa =  transformRangePosToX(this.rangeB);//(this.rangeB-this.a) * (this.getWidth() / interv);
        double ya = this.getHeight()/2;
        return new Rectangle( (int) ( xa - 0.5*this.sizeHandle ) , (int) (ya - 0.5*this.sizeHandle) , this.sizeHandle , this.sizeHandle );
    }

    public JSlimRangeSlider getThis() {return this;}

    @Override
    public void paintComponent(Graphics g) {
        Graphics2D g2 = (Graphics2D) g;
        g2.setStroke(this.strokeNonselectedInterval);
        g2.setPaint(this.paintNonselectedInterval);
        g2.drawLine(0,this.getHeight()/2,this.getWidth(),this.getHeight()/2);

        g2.setStroke(new BasicStroke(sizeHandle/2));
        g2.setPaint(this.paintSelectedInterval);
        double interv = this.b-this.a;
        double xa =  transformRangePosToX(this.rangeA);//(this.rangeA-this.a) * (this.getWidth() / interv);
        double xb =  transformRangePosToX(this.rangeB);//(this.rangeB-this.a) * (this.getWidth() / interv);
        g2.drawLine((int)xa,this.getHeight()/2,(int)xb,this.getHeight()/2);

        g2.setPaint(paintHandleMouseNotOver);
        Rectangle hra = getHandleA();
        Rectangle hrb = getHandleB();
        g2.fillRect(hra.x,hra.y,hra.width,hra.height);
        g2.fillRect(hrb.x,hrb.y,hrb.width,hrb.height);

        g2.setPaint(paintHandleMouseOver);
        if(mouseOverHandle==0) {
            g2.fill(hra);
        }
        if(mouseOverHandle==1) {
            g2.fill(hrb);
        }
    }

    class SlimRangeSliderMouseAndMouseMotionListener implements MouseListener, MouseMotionListener {
        @Override
        public void mouseClicked(MouseEvent e) {
            if( e.getClickCount() == 2) {
                JDialog di = new JDialog();
            }
        }

        @Override
        public void mousePressed(MouseEvent e) {
            int new_handle = isMouseOverHandle(e.getPoint());
            if(new_handle>=0) {
                draggedHandle = new_handle;
                draggingHandle = true;
            }
        }

        @Override
        public void mouseReleased(MouseEvent e) {
            if(draggedHandle>=0) {
                draggingHandle = false;
                draggedHandle  = -1;
            }

            if(eventMode == EventMode.RELEASE_EVENTS) {
                SwingUtilities.invokeLater(new Runnable() {
                    @Override
                    public void run() {
                        //setToolTipText( String.format( "[ %.4f , %.4f ] " , rangeA , rangeB ));
                        fireChangeEvent();
                    }
                });
            }

        }

        @Override
        public void mouseEntered(MouseEvent e) {

        }

        @Override
        public void mouseExited(MouseEvent e) {

        }

        @Override
        public void mouseDragged(MouseEvent e) {
            if(draggingHandle) {
                double interv = b-a;
                if(draggedHandle==0) {
                    double new_ra = transformXToRangePos(e.getX());//a + ( (1.0*e.getX()) / getWidth() ) * interv;
                    new_ra = Math.min(new_ra,rangeB);
                    new_ra = Math.max(new_ra,a);
                    rangeA = new_ra;
                    //System.out.println("ra= "+new_ra);
                }
                else if(draggedHandle==1) {
                    double new_ra = transformXToRangePos(e.getX());//a + ( (1.0*e.getX()) / getWidth() ) * interv;
                    new_ra = Math.max(new_ra,rangeA);
                    new_ra = Math.min(new_ra,b);
                    rangeB = new_ra;
                    //System.out.println("rb= "+new_ra);
                }
            }
            repaint();

            if(eventMode == EventMode.DRAG_EVENTS) {
                SwingUtilities.invokeLater(new Runnable() {
                    @Override
                    public void run() {
                        //setToolTipText( String.format( "[ %.4f , %.4f ] " , rangeA , rangeB ));
                        fireChangeEvent();
                    }
                });
            }
        }

        @Override
        public void mouseMoved(MouseEvent e) {
            if(!draggingHandle) {
                int new_handle = isMouseOverHandle(e.getPoint());
                if(mouseOverHandle != new_handle) {
                    mouseOverHandle = new_handle;
                    repaint();
                }
            }
            else {
                // nothing
            }
        }
    }

    public static void main(String args[]) {
        JFrame fi = new JFrame("Test");
        fi.setSize(600,400);

        JSlimRangeSlider slider = new JSlimRangeSlider(40,200);
        fi.getContentPane().add(slider);
        slider.setSize(300,100);

        fi.setVisible(true);
        fi.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

}