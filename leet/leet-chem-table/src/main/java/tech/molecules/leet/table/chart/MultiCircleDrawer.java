package tech.molecules.leet.table.chart;

import org.jfree.chart.ui.Drawable;

import java.awt.*;
import java.awt.geom.Rectangle2D;
import java.util.List;

public class MultiCircleDrawer implements Drawable {
    /** The outline paint. */
    private List<Paint> outlinePaints;

    /** The outline stroke. */
    private Stroke outlineStroke;

    /** The fill paint. */
    //private Paint fillPaint;

    /**
     * Creates a new instance.
     *
     * @param outlinePaints  the outline paint.
     * @param outlineStroke  the outline stroke.
     */
    public MultiCircleDrawer(List<Paint> outlinePaints,
                             Stroke outlineStroke) {
                            //Paint fillPaint) {
        this.outlinePaints = outlinePaints;
        this.outlineStroke = outlineStroke;
        //this.fillPaint = fillPaint;
    }

    /**
     * Draws the circle.
     *
     * @param g2  the graphics device.
     * @param area  the area in which to draw.
     */
    public void draw(Graphics2D g2, Rectangle2D area) {
        g2.setStroke(this.outlineStroke);
        double currentDegree = 0;
        double degreePerArc  = 360.0/outlinePaints.size();
        // special start settings for 2 and 3 ;)
        if(this.outlinePaints.size()==2) {currentDegree = 90;}
        if(this.outlinePaints.size()==3) {currentDegree = 120;}

        for(Paint pi : this.outlinePaints) {
            g2.setPaint(pi);
            int ax = (int) area.getX(); int ay = (int) area.getY();
            int aw = (int) area.getWidth(); int ah = (int) area.getHeight();
            g2.drawArc(ax,ay,aw,ah, (int)currentDegree ,(int)degreePerArc);
            currentDegree+=degreePerArc;
            currentDegree = currentDegree%360;
        }

//        Ellipse2D ellipse = new Ellipse2D.Double(area.getX(), area.getY(),
//                area.getWidth(), area.getHeight());
//        if (this.fillPaint != null) {
//            g2.setPaint(this.fillPaint);
//            g2.fill(ellipse);
//        }
//        if (this.outlinePaint != null && this.outlineStroke != null) {
//            g2.setPaint(this.outlinePaint);
//            g2.setStroke(this.outlineStroke);
//            g2.draw(ellipse);
//        }
//
//        if(false) {
//            g2.setPaint(Color.black);
//            g2.setStroke(new BasicStroke(1.0f));
//            Line2D line1 = new Line2D.Double(area.getCenterX(), area.getMinY(),
//                    area.getCenterX(), area.getMaxY());
//            Line2D line2 = new Line2D.Double(area.getMinX(), area.getCenterY(),
//                    area.getMaxX(), area.getCenterY());
//            g2.draw(line1);
//            g2.draw(line2);
//        }
    }
}
