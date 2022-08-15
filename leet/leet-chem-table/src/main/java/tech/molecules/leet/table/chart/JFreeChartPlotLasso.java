package tech.molecules.leet.table.chart;

import org.jfree.chart.entity.ChartEntity;
import org.jfree.chart.entity.EntityCollection;
import org.jfree.chart.entity.StandardEntityCollection;

import java.awt.geom.Path2D;
import java.util.Collection;

/**
 * Implements lasso select.  USE ONLY WITH PLOT DIAGRAMS.
 */
public class JFreeChartPlotLasso {

    /** Representing the enclosed area*/
    private Path2D poly;
    /** To later hold all ChartEntitys within the polygon after they've been
     * calculated once, to not have to search them again. */
    private EntityCollection ec = null;

    public JFreeChartPlotLasso(double x, double y) {
        poly = new Path2D.Double();
        poly.moveTo(x,y);
    }

    /**
     *
     * @param entityCollection containing all ChartEntity-Objects to be searched through
     * @return EntityCollection containing all ChartEntity-Object within the borders of the selection
     */
    public EntityCollection getContainedEntities(EntityCollection entityCollection) {
        if (ec == null) {
            ec = new StandardEntityCollection();
            Collection entities = entityCollection.getEntities();
            for (int i = 0; i < entities.size(); i++) {
                ChartEntity entity = entityCollection.getEntity(i);
                //if( entity.getArea().intersects(poly.getBounds()) ) {
                //System.out.println("entity: "+entity.getArea());
                if( poly.intersects( entity.getArea().getBounds() ) ) {
                    ec.add( entity );
                }
            }
        }
        return ec;
    }

    /**
     * Returns the previously calculated entities.  If
     * <code>getContainedEntitys(EntityCollection entityCollection)</code>
     * hasn't been called beforehand, it'll return <code>null</code>
     *
     * @return EntityCollection
     */
    public EntityCollection getContainedEntities() {
        return ec;
    }
//
//    /**
//     * Reads out all diagram-coordinates of the points.
//     *
//     * @param entityCollection containing all ChartEntity-Objects to be searched through
//     * @return ArrayList containing
//     */
//    public HashMap<EvokerPoint2D, String> getContainedPointsInd(EntityCollection entityCollection) {
//        HashMap<EvokerPoint2D, String> hm_ret = new HashMap<EvokerPoint2D, String>();
//        getContainedEntitys(entityCollection);
//        Collection entities = ec.getEntities();
//        for (int i = 0; i < entities.size(); i++) {
//            ChartEntity entity = ec.getEntity(i);
//            hm_ret.put(getCoordinatesOfEntity(entity), getIndOfEntity(entity));
//        }
//        return hm_ret;
//    }


    /**
     * Adds a point to the polygon
     * @param x coordinate
     * @param y coordinate
     */
    public void addPoint(double x, double y) {
        poly.lineTo(x,y);
    }

    public void close(){
        poly.closePath();
    }

    public Path2D getPath() {
        return this.poly;
    }

}