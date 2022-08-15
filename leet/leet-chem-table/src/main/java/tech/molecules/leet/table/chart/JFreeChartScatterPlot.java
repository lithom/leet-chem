package tech.molecules.leet.table.chart;

import net.mahdilamb.colormap.Colormap;
import org.apache.commons.lang3.tuple.Pair;
import org.jfree.chart.*;
import org.jfree.chart.annotations.*;
import org.jfree.chart.entity.ChartEntity;
import org.jfree.chart.entity.EntityCollection;
import org.jfree.chart.entity.StandardEntityCollection;
import org.jfree.chart.entity.XYItemEntity;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.PaintScale;
import org.jfree.chart.renderer.xy.XYShapeRenderer;
import org.jfree.data.general.DatasetChangeListener;
import org.jfree.data.general.DatasetGroup;
import org.jfree.data.general.DefaultKeyedValuesDataset;
import org.jfree.data.general.KeyedValuesDataset;
import tech.molecules.leet.table.NColumn;
import tech.molecules.leet.table.NDataProvider;
import tech.molecules.leet.table.NexusTableModel;
import tech.molecules.leet.table.NumericalDatasource;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.geom.*;
import java.util.List;
import java.util.*;
import java.util.stream.Collectors;


/**
 * Provides synchronization in between NexusTableModel and scatter plot.
 *
 * Data can be provided in different ways into the plot.
 * First option is, to have fixed x/y coordinates for every row in the
 * dataset.
 *
 */
public class JFreeChartScatterPlot extends JPanel {

    //private JPanel chart;
    private ChartPanel cp;

    private NexusTableModel ntm;
    //private Map<String,double[]> dataXY;
    //private Map<String,Double>   dataColor;
    private KeyedValuesDataset  dataX;
    private KeyedValuesDataset  dataY;
    private DefaultKeyedValuesDataset  dataCol;
    double coloringLB = 0;
    double coloringUB = 1;

    private Set<String>          dataHighlight;
    private Set<String>          dataSelection;
    //private Set<String>          dataHighlightBackup; // necessary for the context menu..


    public JFreeChartScatterPlot(NexusTableModel ntm, KeyedValuesDataset x, KeyedValuesDataset y) {
        this.ntm = ntm;
        this.dataX = x;
        this.dataY = y;
        this.updatePlot();
        //this.initContextMenu(new ArrayList<>(),new ArrayList<>());
    }

    public JFreeChartScatterPlot(NexusTableModel ntm,  NDataProvider dp_x , NDataProvider dp_y, NumericalDatasource nd_x, NumericalDatasource nd_y) {
        this.ntm = ntm;
        this.dataX = new NumericalDatasourceKeyedDataset(ntm,nd_x,dp_x);
        this.dataY = new NumericalDatasourceKeyedDataset(ntm,nd_y,dp_y);
        this.updatePlot();
    }

    public void setColor(DefaultKeyedValuesDataset data_color) {
        this.dataCol = data_color;
        Map<String,Double> cold = new HashMap<>();
        for(String si : this.ntm.getAllRows()) {cold.put(si,data_color.getValue(si).doubleValue());}
        this.setColorValues(cold);
    }

    private Colormap colormap;

    public void setColormap(Colormap colormap) {
        this.colormap = colormap;
        this.setColor(this.dataCol);
    }

    public double[] getPositionOfKey(String ski) {

        int ki = getIndexForKey(ski);
        double kpx = this.cp.getChartRenderingInfo().getEntityCollection().getEntity( ki ).getArea().getBounds2D().getCenterX();
        double kpy = this.cp.getChartRenderingInfo().getEntityCollection().getEntity( ki ).getArea().getBounds2D().getCenterY();
        return new double[] { kpx, kpy };
    }

    private boolean isMouseContextMenuShown = false;

    private List<XYAnnotation> highlightAnnotations = new ArrayList<>();
    private List<XYAnnotation> selectionAnnotations = new ArrayList<>();
    private List<XYAnnotation> multiClassAnnotations = new ArrayList<>();

    /**
     * Note: calling this method does NOT fire any events in ClusterAppModel
     *
     *
     * @param selection
     */
    public void setSelection(Set<String> selection) {
        this.dataSelection = new HashSet<>(selection);
        //Map<String,Double> hm = getHighlightMap(highlight);//getHighlightMap(all_e,ec);
        //setColorValues(hm);

        selectionAnnotations.stream().forEach( ai -> this.cp.getChart().getXYPlot().removeAnnotation(ai) );
        Set<Integer> hix = new HashSet<>(selection.stream().map( hi -> getIndexForKey(hi) ).collect(Collectors.toList()));

        selectionAnnotations.clear();
        for(int hi : hix) {
            double px = data.getX(0,hi).doubleValue();
            double py = data.getY(0,hi).doubleValue();
            System.out.println("selection: "+px+" / "+py+" idx="+hi);
            CircleDrawer circleDrawer = new CircleDrawer(Color.cyan,new BasicStroke(2),new Color(50,160,240,140));
            XYAnnotation hlan = new XYDrawableAnnotation(px,py,16,16,circleDrawer);
            selectionAnnotations.add(hlan);
            this.cp.getChart().getXYPlot().addAnnotation(hlan,true);
            //}
        }
    }

    public void setHighlight(Set<String> highlight, boolean fireEvent) {
        this.dataHighlight = highlight;
        //Map<String,Double> hm = getHighlightMap(highlight);//getHighlightMap(all_e,ec);
        //setColorValues(hm);

        highlightAnnotations.stream().forEach( ai -> this.cp.getChart().getXYPlot().removeAnnotation(ai) );
        Set<Integer> hix = new HashSet<>(highlight.stream().map( hi -> getIndexForKey(hi) ).collect(Collectors.toList()));

        highlightAnnotations.clear();
        for(int hi : hix) {
            //ChartEntity ce = this.cp.getChartRenderingInfo().getEntityCollection().getEntity(hi);
            //if(ce instanceof XYItemEntity) {
            //    XYItemEntity cex = (XYItemEntity) ce;
                //double px = cex.getArea().getBounds().getCenterX();
                //double py = cex.getArea().getBounds().getCenterY();
                double px = data.getX(0,hi).doubleValue();
                double py = data.getY(0,hi).doubleValue();
                System.out.println("highlight: "+px+" / "+py+" idx="+hi);
                CircleDrawer circleDrawer = new CircleDrawer(Color.red,new BasicStroke(2),new Color(214,120,120,120));
                XYAnnotation hlan = new XYDrawableAnnotation(px,py,16,16,circleDrawer);
                highlightAnnotations.add(hlan);
                this.cp.getChart().getXYPlot().addAnnotation(hlan,true);
            //}
        }



        if(fireEvent) {
            //fireHighlightingChangedEvent(new NexusHighlightingChangedEvent(this, highlight));
        }
        //this.updatePlot();
    }



    /**
     *
     * @param classes
     * @param paints
     */
    public void setMultipleAnnotations(Map<String,List<Integer>> classes, Map<Integer,Paint> paints ) {
        multiClassAnnotations.stream().forEach( ai -> this.cp.getChart().getXYPlot().removeAnnotation(ai) );
        multiClassAnnotations = new ArrayList<>();

        Set<Pair<String,Integer>> hix = new HashSet<>(classes.keySet().stream().map( hi -> Pair.of(hi,getIndexForKey(hi)) ).collect(Collectors.toList()));
        for( Pair<String,Integer> pii : hix) {
            int hi = pii.getRight();
            String struc = pii.getLeft();
            double px = data.getX(0,hi).doubleValue();
            double py = data.getY(0,hi).doubleValue();
            System.out.println("multiclass: "+px+" / "+py+" idx="+hi);
            List<Paint> paints_i = classes.get(struc).stream().map( si -> paints.get(si) ).collect(Collectors.toList());
            MultiCircleDrawer circleDrawer = new MultiCircleDrawer(paints_i,new BasicStroke(2));
            XYAnnotation hlan = new XYDrawableAnnotation(px,py,16,16,circleDrawer);
            multiClassAnnotations.add(hlan);
            this.cp.getChart().getXYPlot().addAnnotation(hlan,true);
            //}
        }
    }


    private Map<String,Double> lastColorValues;

    public void setColorValues(Map<String,Double> col) {
        double c_min = Double.POSITIVE_INFINITY;
        double c_max = Double.NEGATIVE_INFINITY;
        //Random ri = new Random();
        //for(String ci : col.keySet()) { // no, we should color all rows, not assigned ones get a nan..
        for(String ci : this.ntm.getAllRows()) {
            Double cvi = col.get(ci);
            if(cvi==null) {cvi = Double.NaN;}
            System.out.println("v: "+cvi);
            this.dataCol.setValue(ci,cvi);
            if( Double.isFinite( cvi ) ) {
                c_min = Math.min(cvi,c_min);
                c_max = Math.max(cvi,c_max);
            }

        }

        double paintscale_min = c_min-Math.max(0.001, (c_max-c_min)*0.01 );
        double paintscale_max = c_max+Math.max(0.001, (c_max-c_min)*0.01 );
        //((XYShapeRenderer)this.cp.getChart().getXYPlot().getRenderer()).setPaintScale(new SpectrumPaintScale(paintscale_min,paintscale_max));


        Colormap cm = net.mahdilamb.colormap.Colormaps.get("Jet");
        if(this.colormap!=null) {cm = this.colormap;}

        PaintScaleFromColormap psfc = new PaintScaleFromColormap(cm,paintscale_min,paintscale_max,0.75, new Color(180,200,210,80));
        ((XYShapeRenderer)this.cp.getChart().getXYPlot().getRenderer()).setPaintScale(psfc);
        //this.data.
        //this.data.fireDatasetChangedExplicitly();
        this.data.setZ(this.dataCol);
        //this.lastColorValues =

        //PaintScaleLegend psl = new PaintScaleLegend(psfc,new NumberAxis("Value"));
        //psl.setPosition(RectangleEdge.RIGHT);
        //this.cp.getChart().addSubtitle(psl);
    }

    private XYChartCreator.CombinedKeyedXYZDataset<String> data;


    public String getKeyForIndex(int i) {
        return (String) this.data.getKey(i);
    }

    public int getIndexForKey(String key) {
        return this.data.getIndex(key);
    }

    public void setColoringByNexusColumn(NColumn nc) {
        Map<String,Double> col_values = new HashMap<>();
        for(String si : this.ntm.getAllRows()) {
            col_values.put( si , nc.evaluateNumericalDataSource(ntm.getDatasetForColumn(nc), "", si) );
        }
        setColorValues(col_values);
    }

    public class SetClusteringAnnotationsAction extends AbstractAction {
        private List<Pair<Color,List<String>>> clustering;

        public SetClusteringAnnotationsAction(String name, List<Pair<Color, List<String>>> clustering) {
            super(name);
            this.clustering = clustering;
        }

        @Override
        public void actionPerformed(ActionEvent e) {
            Map<Integer,Paint> paints = new HashMap<>();
            Map<String,List<Integer>> clustered = new HashMap<>();
            for(int zi=0;zi<clustering.size();zi++) {
                Pair<Color,List<String>> pci = this.clustering.get(zi);
                paints.put(zi,pci.getLeft());
                for(String struci : pci.getRight()) {
                    if(!clustered.containsKey(struci)) {clustered.put(struci,new ArrayList<>());}
                    clustered.get(struci).add(zi);
                }
            }
            setMultipleAnnotations(clustered,paints);
        }
    }

    public class SetColoringAction extends AbstractAction{
        private Map<String,Double> nd;
        public SetColoringAction(String name, Map<String,Double> data) {
            super(name);
            this.nd = data;
        }

        @Override
        public void actionPerformed(ActionEvent e) {
            setColorValues(this.nd);
        }
    }

    public class SetColormapAction extends AbstractAction {
        private Colormap cm;
        public SetColormapAction(String name, Colormap cm) {
            super(name);
            this.cm = cm;
        }
        @Override
        public void actionPerformed(ActionEvent e) {
            setColormap(this.cm);
        }
    }

//    public void initContextMenu(List<Action> global_coloring, List<Action> global_colormap) {
//        JPopupMenu jpop = new JPopupMenu();
//        jpop.add(new JMenuItem("Select"));
//        JMenu jm_coloring = new JMenu("Set coloring");
//        for(Action aci : global_coloring) { jm_coloring.add(aci); }
//
//        JMenu jm_colormap = new JMenu("Set colormap");
//        for(Action aci : global_colormap) {jm_colormap.add(aci);}
//        //cms.stream().forEach( ci -> jm_colormap.add( new SetColormapAction( ci.getLeft() , ci.getRight()) ) );
//
//        jpop.add(jm_coloring);
//        jpop.add(jm_colormap);
//        this.cp.setPopupMenu(jpop);
//    }

    public void setContextMenu(JPopupMenu jpop) {
        this.cp.setPopupMenu(jpop);
    }


    public String getKeyForXYEntity(XYItemEntity entity) {
        return this.data.getKey( entity.getItem() );
    }

    public void updatePlot() {
        KeyedValuesDataset px = this.dataX;//new DefaultKeyedValuesDataset();
        KeyedValuesDataset py = this.dataY;//new DefaultKeyedValuesDataset();
        KeyedValuesDataset cc = this.dataCol;//new DefaultKeyedValuesDataset();

        if(cc==null) {
            Random rand = new Random();
            DefaultKeyedValuesDataset ccn = new DefaultKeyedValuesDataset();
            for(String ri : this.ntm.getAllRows()) { ccn.setValue(ri,rand.nextDouble());}
            cc = ccn;
            this.dataCol = ccn;
        }

        this.data = new XYChartCreator.CombinedKeyedXYZDataset<>(px, py, cc);

        //JFreeChart chart = ChartFactory.createScatterPlot("test","x","y", data);
        JFreeChart chart = ChartFactory.createScatterPlot(null,null,null, data);
        chart.removeLegend();
        chart.getXYPlot().setBackgroundPaint(Color.black);
        chart.setBackgroundPaint(Color.black);
        chart.getXYPlot().setDomainGridlinesVisible(false);
        chart.getXYPlot().setRangeGridlinesVisible(false);
        chart.getXYPlot().getDomainAxis().setVisible(false);
        chart.getXYPlot().getRangeAxis().setVisible(false);

        chart.setBorderVisible(true);
        chart.setBorderPaint(Color.orange.darker());
        chart.setBorderStroke(new BasicStroke(2));


        //((XYLineAndShapeRenderer)(chart.getXYPlot().getRenderer()).;
        XYShapeRenderer renderer = new XYShapeRenderer();
        renderer.setPaintScale(new SpectrumPaintScale(0,1));

        chart.getXYPlot().setRenderer(renderer);

        this.cp = new ChartPanel(chart);

        this.removeAll();
        this.setLayout(new BorderLayout());
        this.add(cp,BorderLayout.CENTER);

        this.cp.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseExited(MouseEvent e) {
                // clear the highlighting annotations, if we really leave the component, also
                // based on x/y coordinates.
                if(!contains(e.getX(),e.getY()))
                {
                    setHighlight(new HashSet(), true);
                }

                //}
            }
            @Override
            public void mouseEntered(MouseEvent e) {

            }
        });
        cp.addChartMouseListener(new ChartMouseListener() {
            @Override
            public void chartMouseClicked(ChartMouseEvent chartMouseEvent) {
                if(chartMouseEvent.getTrigger().getButton()==MouseEvent.BUTTON3) {
                    isMouseContextMenuShown = true;
                }
            }

            @Override
            public void chartMouseMoved(ChartMouseEvent chartMouseEvent) {
                ChartEntity ent = chartMouseEvent.getEntity();
                if( ent != null) {
                    if( ent instanceof XYItemEntity) {
                        System.out.println("Entity: " + ent.toString() );
                    }
                }

                if(highlightNNearestNeighbors>=1) {
                    //processHighlightNNearestNeighbors(  );
                    EntityCollection all_e = cp.getChartRenderingInfo().getEntityCollection();
                    EntityCollection ec = getNNearestEntities(all_e,chartMouseEvent.getTrigger().getX(),chartMouseEvent.getTrigger().getY(),highlightNNearestNeighbors);
                    System.out.println("to highlight: ");
                    for(Object cei : ec.getEntities()) {
                        System.out.println("nearest: "+cei.toString());
                    }
                    List<String> h_keys = (List<String>) ec.getEntities().stream().map(ei -> getKeyForXYEntity( (XYItemEntity) ei)).collect(Collectors.toList());
                    setHighlight( new HashSet(h_keys) , true );
                }
            }
        });

        LassoMouseListener lassoML = new LassoMouseListener();
        cp.addChartMouseListener(lassoML);
    }


    private int highlightNNearestNeighbors = 6;

    /**
     * set to -1 to deactivate
     * @param n
     */
    public void setHighlightNNearestNeighbors(int n) {
        this.highlightNNearestNeighbors = n;
    }


    public static double[] java2dToChart(ChartPanel cp, XYPlot plot,double px, double py) {
        Rectangle2D plotArea = cp.getScreenDataArea();
        double chartX = plot.getDomainAxis().java2DToValue(px, plotArea, plot.getDomainAxisEdge());
        double chartY = plot.getRangeAxis().java2DToValue(py, plotArea, plot.getRangeAxisEdge());
        return new double[]{chartX,chartY};
    }


    public Map<String,Double> getHighlightMap(Collection<String> highlight) {
        Map<String,Double> hmap = new HashMap<>();
        for(String ri : this.ntm.getVisibleRows()) {
            hmap.put( ri , highlight.contains(ri)?1.0:-1.0 );
        }
        return hmap;
    }

        public Map<String,Double> getHighlightMap(EntityCollection entityCollection, EntityCollection toHighlight) {
        Map<String,Double> hmap = new HashMap<>();
        for( int zi=0;zi<entityCollection.getEntityCount();zi++) {
            if(entityCollection.getEntity(zi) instanceof XYItemEntity) {
                 String ki = getKeyForXYEntity((XYItemEntity)entityCollection.getEntity(zi));
                 if(toHighlight.getEntities().contains(entityCollection.getEntity(zi))) {
                     hmap.put(ki,1.0);
                 }
                 else {
                     hmap.put(ki,0.0);
                 }
            }
        }
        return hmap;
    }

    public static EntityCollection getNNearestEntities(EntityCollection entityCollection, int px, int py, int n) {

        EntityCollection ec = new StandardEntityCollection();
        Collection entities = entityCollection.getEntities();
        List<Pair<Double,XYItemEntity>> entitiesWithD = new ArrayList<>();
        for (int i = 0; i < entities.size(); i++) {
            if( ! (entityCollection.getEntity(i) instanceof XYItemEntity) ) {continue;}
            XYItemEntity entity = (XYItemEntity) entityCollection.getEntity(i);
            //if( entity.getArea().intersects(poly.getBounds()) ) {
            //System.out.println("entity: "+entity.getArea());
            double cx = entity.getArea().getBounds().getCenterX();
            double cy = entity.getArea().getBounds().getCenterY();

            entitiesWithD.add( Pair.of(  (px-cx)*(px-cx) + (py-cy)*(py-cy)  , entity) );
        }
        entitiesWithD.sort( (x,y) -> Double.compare( x.getLeft() , y.getLeft() ) );
        for(int zi=0;zi<n;zi++) {
            ec.add( entitiesWithD.get(zi).getRight() );
        }
        return ec;
    }

    public static interface ScatterPlotListener {
        public void highlightingChanged(NexusTableModel.NexusHighlightingChangedEvent e);
        public void selectionChanged(NexusTableModel.NexusSelectionChangedEvent e);
    }

    List<ScatterPlotListener> listeners = new ArrayList<>();

    public void addScatterPlotListener(ScatterPlotListener li) {
        this.listeners.add(li);
    }
    public void removeScatterPlotListener(ScatterPlotListener li) {
        this.listeners.remove(li);
    }

    private void fireHighlightingChangedEvent(NexusTableModel.NexusHighlightingChangedEvent event) {
        for(ScatterPlotListener li : listeners) {
            li.highlightingChanged(event);
        }
    }

    public enum LASSO_STATE {NONE,DRAGGING};

    public class LassoMouseListener implements ChartMouseListener {


        private LASSO_STATE state = LASSO_STATE.NONE;

        JFreeChartPlotLasso lasso = null;

        XYPolygonAnnotation lassoAnnotation = null;//new XYPolygonAnnotation();

        @Override
        public void chartMouseClicked(ChartMouseEvent chartMouseEvent) {

            if( chartMouseEvent.getTrigger().isShiftDown() && state == LASSO_STATE.NONE ) {
                //double xy[] = getXY(chartMouseEvent);
                double xy[] = new double[]{ chartMouseEvent.getTrigger().getPoint().x , chartMouseEvent.getTrigger().getPoint().y };
                state = LASSO_STATE.DRAGGING;
                this.lasso =  new JFreeChartPlotLasso(xy[0],xy[1]);
                System.out.println("start drag..");

                Path2D lpath = (Path2D) this.lasso.getPath().clone();
                if(this.lassoAnnotation!=null) {
                    chartMouseEvent.getChart().getXYPlot().removeAnnotation(this.lassoAnnotation);
                }
                this.lassoAnnotation = createAnnotationFromPath(lpath);
                chartMouseEvent.getChart().getXYPlot().addAnnotation(this.lassoAnnotation);
            }
            else if(state == LASSO_STATE.DRAGGING ) {
                state = LASSO_STATE.NONE;

                // close lasso, then report selection:
                this.lasso.close();
                EntityCollection all_e = cp.getChartRenderingInfo().getEntityCollection();
                EntityCollection ec = this.lasso.getContainedEntities(all_e);
                System.out.println("selected entities: "+ec.getEntities().size());

                chartMouseEvent.getChart().getXYPlot().removeAnnotation(this.lassoAnnotation);
                this.lassoAnnotation = null;
            }
        }

        public double[] dumpPathIteratorCoords(PathIterator pathIterator) {
            List<Double> path = new ArrayList<>();
            float[] coords = new float[6];
            //PathIterator pathIterator = shape.getPathIterator(new AffineTransform());
            while (!pathIterator.isDone()) {
                switch (pathIterator.currentSegment(coords)) {
                    case PathIterator.SEG_MOVETO:
                        //System.out.printf("move to x1=%f, y1=%f\n",
                        //        coords[0], coords[1]);
                        path.add((double)coords[0]);
                        path.add((double)coords[1]);
                        break;
                    case PathIterator.SEG_LINETO:
                        //System.out.printf("line to x1=%f, y1=%f\n",
                        //        coords[0], coords[1]);
                        path.add((double)coords[0]);
                        path.add((double)coords[1]);
                        break;
                    case PathIterator.SEG_QUADTO:
                        //System.out.printf("quad to x1=%f, y1=%f, x2=%f, y2=%f\n",
                        //        coords[0], coords[1], coords[2], coords[3]);
                        break;
                    case PathIterator.SEG_CUBICTO:
                        //System.out.printf("cubic to x1=%f, y1=%f, x2=%f, y2=%f, x3=%f, y3=%f\n",
                        //        coords[0], coords[1], coords[2], coords[3], coords[4], coords[5]);
                        break;
                    case PathIterator.SEG_CLOSE:
                        //System.out.printf("close\n");
                        break;
                }
                pathIterator.next();
            }
            return path.stream().mapToDouble(xi->xi).toArray();
        }

        private XYPolygonAnnotation createAnnotationFromPath(Path2D path) {
            path.closePath();
            PathIterator pi = path.getPathIterator(AffineTransform.getScaleInstance(1,1));
            //double[] points = new double[]{1,1,1,3,3,3,3,1};
            double[] points = dumpPathIteratorCoords(pi);

            // transform to chart..
            double[] points_chart = new double[points.length];
            for(int zi=0;zi<points_chart.length/2;zi++) {
                double pxi[] = java2dToChart(cp,cp.getChart().getXYPlot(),points[2*zi+0],points[2*zi+1]);
                points_chart[zi*2+0] = pxi[0];
                points_chart[zi*2+1] = pxi[1];
            }

            XYPolygonAnnotation poly_annotation = new XYPolygonAnnotation(points_chart);
            return poly_annotation;
        }

        public double[] getXY(ChartMouseEvent chartMouseEvent) {
            Point2D p = cp.translateScreenToJava2D(chartMouseEvent.getTrigger().getPoint());
            Rectangle2D plotArea = cp.getScreenDataArea();
            XYPlot plot = (XYPlot) chartMouseEvent.getChart().getPlot(); // your plot
            double chartX = plot.getDomainAxis().java2DToValue(p.getX(), plotArea, plot.getDomainAxisEdge());
            double chartY = plot.getRangeAxis().java2DToValue(p.getY(), plotArea, plot.getRangeAxisEdge());
            return new double[]{chartX,chartY};
        }



        @Override
        public void chartMouseMoved(ChartMouseEvent chartMouseEvent) {
            if(state == LASSO_STATE.DRAGGING) {
                //double pxy[] = getXY(chartMouseEvent);
                double pxy[] = new double[]{ chartMouseEvent.getTrigger().getPoint().x , chartMouseEvent.getTrigger().getPoint().y };
                //System.out.println("add: "+pxy[0]+" "+pxy[1]);
                this.lasso.addPoint( pxy[0] , pxy[1] );

                Path2D lpath = (Path2D) this.lasso.getPath().clone();
                chartMouseEvent.getChart().getXYPlot().removeAnnotation(this.lassoAnnotation);
                this.lassoAnnotation = createAnnotationFromPath(lpath);
                chartMouseEvent.getChart().getXYPlot().addAnnotation(this.lassoAnnotation);

            }
        }
    }


    private static class PaintScaleFromColormap implements PaintScale {

        private Colormap cm;
        private double lb, ub;
        double transparency;
        Color nanColor;

        public PaintScaleFromColormap(Colormap cm, double lb, double ub, double transparency, Color nanColor) {
            this.cm = cm;
            this.lb = lb; this.ub = ub;
            this.transparency = transparency;
            this.nanColor = nanColor;
        }

        @Override
        public double getLowerBound() {
            return this.lb;
        }

        @Override
        public double getUpperBound() {
            return this.ub;
        }

        @Override
        public Paint getPaint(double v) {
            if(Double.isNaN(v)) {
                if(this.nanColor!=null) {
                    return this.nanColor;
                    //return new Color( this.nanColor.getRed() , this.nanColor.getGreen() , this.nanColor.getBlue() , (int)(255*this.transparency) )
                }
            }
            double va = (v-lb)/(ub-lb);
            //return cm.get( va );
            Color ca = cm.get( va );
            return new Color(ca.getRed(),ca.getGreen(),ca.getBlue(), (int)(255*this.transparency) );
        }
    }

    private static class SpectrumPaintScale implements PaintScale {

        private static final float H1 = 0.25f;
        private static final float H2 = 0.75f;
        private final double lowerBound;
        private final double upperBound;

        public SpectrumPaintScale(double lowerBound, double upperBound) {
            this.lowerBound = lowerBound;
            this.upperBound = upperBound;
        }

        @Override
        public double getLowerBound() {
            return lowerBound;
        }

        @Override
        public double getUpperBound() {
            return upperBound;
        }

        @Override
        public Paint getPaint(double value) {
            float scaledValue = (float) (value / (getUpperBound() - getLowerBound()));
            float scaledH = H1 + scaledValue * (H2 - H1);
            return Color.getHSBColor(scaledH, 1f, 1f);
        }
    }


    public static class NumericalDatasourceKeyedDataset<U> implements KeyedValuesDataset {
        private NexusTableModel ntm;
        private NumericalDatasource<U>  nds;
        private U dp;

        public NumericalDatasourceKeyedDataset(NexusTableModel ntm, NumericalDatasource<U> nds, U dp) {
            this.ntm = ntm;
            this.nds = nds;
            this.dp = dp;
            reinit();
        }

        private void reinit() {
            //this.ntm.getVisibleRows()
        }

        @Override
        public Comparable getKey(int i) {
            return ntm.getVisibleRows().get(i);
        }

        @Override
        public int getIndex(Comparable comparable) {
            return ntm.getVisibleRows().indexOf(comparable);
        }

        @Override
        public List getKeys() {
            return ntm.getVisibleRows();
        }

        @Override
        public Number getValue(Comparable comparable) {
            if( nds.hasValue(dp,(String)comparable) ) {
                return nds.getValue(dp,(String) comparable);
            }
            return null;
        }

        @Override
        public int getItemCount() {
            return ntm.getVisibleRows().size();
        }

        @Override
        public Number getValue(int i) {
            String ri = ntm.getVisibleRows().get(i);
            if( nds.hasValue(dp,ri) ) {
                return nds.getValue(dp,ri);
            }
            return null;
        }

        private List<DatasetChangeListener> listeners = new ArrayList<>();

        @Override
        public void addChangeListener(DatasetChangeListener datasetChangeListener) {
            this.listeners.add(datasetChangeListener);
        }

        @Override
        public void removeChangeListener(DatasetChangeListener datasetChangeListener) {
            this.listeners.remove(datasetChangeListener);
        }

        private DatasetGroup group;
        @Override
        public DatasetGroup getGroup() {
            return this.group;
        }

        @Override
        public void setGroup(DatasetGroup datasetGroup) {
            this.group = datasetGroup;
        }
    }
}
