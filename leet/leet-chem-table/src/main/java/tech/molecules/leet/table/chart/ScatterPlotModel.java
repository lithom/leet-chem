package tech.molecules.leet.table.chart;

import net.mahdilamb.colormap.Colormap;
import org.apache.commons.lang3.tuple.Pair;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.annotations.XYAnnotation;
import org.jfree.chart.annotations.XYDrawableAnnotation;
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
import org.knowm.xchart.style.markers.Circle;
import tech.molecules.leet.table.NDataProvider;
import tech.molecules.leet.table.NexusTableModel;
import tech.molecules.leet.table.NumericalDatasource;
import tech.molecules.leet.util.ColorMapHelper;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Rectangle2D;
import java.util.*;
import java.util.List;
import java.util.stream.Collectors;

public class ScatterPlotModel {

    private NexusTableModel ntm;
    //private Map<String,double[]> dataXY;
    //private Map<String,Double>   dataColor;
    private KeyedValuesDataset dataX;
    private KeyedValuesDataset  dataY;
    private DefaultKeyedValuesDataset dataCol;
    double coloringLB = 0;
    double coloringUB = 1;

    private Set<String> dataHighlight;
    private Set<String> dataSelection;

    private boolean isMouseContextMenuShown = false;

    private List<XYAnnotation> highlightAnnotations = new ArrayList<>();
    private List<XYAnnotation> selectionAnnotations = new ArrayList<>();
    private List<XYAnnotation> multiClassAnnotations = new ArrayList<>();

    private XYShapeRenderer renderer;

    public ScatterPlotModel(NexusTableModel ntm, KeyedValuesDataset x, KeyedValuesDataset y) {
        this.ntm = ntm;
        this.dataX = x;
        this.dataY = y;
        this.reinitPlot();
        //this.initContextMenu(new ArrayList<>(),new ArrayList<>());
    }

    public ScatterPlotModel(NexusTableModel ntm, NumericalDatasource nd_x, NumericalDatasource nd_y) {
        this.ntm = ntm;
        this.dataX = new JFreeChartScatterPlot.NumericalDatasourceKeyedDataset(ntm,nd_x);
        this.dataY = new JFreeChartScatterPlot.NumericalDatasourceKeyedDataset(ntm,nd_y);
        //this.updatePlot();
        this.reinitPlot();
    }

    public boolean setMouseOverClass(NexusTableModel.SelectionType st) {
        return this.ntm.registerSelectionType(st);
    }

    public void setColor(DefaultKeyedValuesDataset data_color) {
        this.dataCol = data_color;
        Map<String,Double> cold = new HashMap<>();
        for(String si : this.ntm.getAllRows()) {cold.put(si,data_color.getValue(si).doubleValue());}
        this.setColorValues(cold);
    }

    public void setColorExpclicit(PaintScale colors, Map<String,Integer> values) {
        ((XYShapeRenderer)this.getChart().getXYPlot().getRenderer()).setPaintScale(colors);
        for(String si : this.ntm.getAllRows()) {
            this.dataCol.setValue(si,values.get(si));
        }
        this.data.setZ(this.dataCol);
    }

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

        ColorMapHelper.PaintScaleFromColormap psfc = new ColorMapHelper.PaintScaleFromColormap(cm,paintscale_min,paintscale_max,0.75, new Color(180,200,210,80));
        //((XYShapeRenderer)this.cp.getChart().getXYPlot().getRenderer()).setPaintScale(psfc);
        ((XYShapeRenderer)this.getChart().getXYPlot().getRenderer()).setPaintScale(psfc);


        //this.data.
        //this.data.fireDatasetChangedExplicitly();
        this.data.setZ(this.dataCol);
        //this.lastColorValues =

        //PaintScaleLegend psl = new PaintScaleLegend(psfc,new NumberAxis("Value"));
        //psl.setPosition(RectangleEdge.RIGHT);
        //this.cp.getChart().addSubtitle(psl);
    }

    public void setColorValues(NumericalDatasource nds) {
        Map<String,Double> cvalues = new HashMap<>();
        for(String ri : ntm.getAllRows()) {
            if( nds.hasValue(ri) ) {
                cvalues.put(ri,nds.getValue(ri));
            }
        }
        setColorValues(cvalues);
    }

    public NexusTableModel getNexusTableModel() {
        return this.ntm;
    }

    private Colormap colormap;

    public void setColormap(Colormap colormap) {
        this.colormap = colormap;
        this.setColor(this.dataCol);
    }

/*
    public double[] getPositionOfKey(String ski) {
        int ki = getIndexForKey(ski);
        double kpx = this.cp.getChartRenderingInfo().getEntityCollection().getEntity( ki ).getArea().getBounds2D().getCenterX();
        double kpy = this.cp.getChartRenderingInfo().getEntityCollection().getEntity( ki ).getArea().getBounds2D().getCenterY();
        return new double[] { kpx, kpy };
    }
*/

    public String getKeyForXYEntity(XYItemEntity entity) {
        return this.data.getKey( entity.getItem() );
    }


    private int highlightNNearestNeighbors = 6;

    /**
     * set to -1 to deactivate
     * @param n
     */
    public void setHighlightNNearestNeighbors(int n) {
        this.highlightNNearestNeighbors = n;
    }


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

        selectionAnnotations.stream().forEach( ai -> this.getXYPlot().removeAnnotation(ai) );
        Set<Integer> hix = new HashSet<>(selection.stream().map( hi -> getIndexForKey(hi) ).collect(Collectors.toList()));

        selectionAnnotations.clear();
        for(int hi : hix) {
            double px = data.getX(0,hi).doubleValue();
            double py = data.getY(0,hi).doubleValue();
            System.out.println("selection: "+px+" / "+py+" idx="+hi);
            CircleDrawer circleDrawer = new CircleDrawer(Color.cyan,new BasicStroke(2),new Color(50,160,240,140));
            XYAnnotation hlan = new XYDrawableAnnotation(px,py,16,16,circleDrawer);
            selectionAnnotations.add(hlan);
            this.getXYPlot().addAnnotation(hlan,true);
            //}
        }
    }

    public void setHighlight(Set<String> highlight, boolean fireEvent) {
        this.dataHighlight = highlight;
        //Map<String,Double> hm = getHighlightMap(highlight);//getHighlightMap(all_e,ec);
        //setColorValues(hm);
        highlightAnnotations.stream().forEach( ai -> this.chart.getXYPlot().removeAnnotation(ai) );
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
            this.getXYPlot().addAnnotation(hlan,true);
            //}
        }



        if(fireEvent) {
            fireHighlightingChangedEvent(new NexusTableModel.NexusHighlightingChangedEvent(this, highlight));
        }
        //this.updatePlot();
    }


    public static double[] java2dToChart(ChartPanel cp, XYPlot plot, double px, double py) {
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

    public int getHighlightNNearestNeighbors() {
        return this.highlightNNearestNeighbors;
    }

    public static interface ScatterPlotListener {
        public void highlightingChanged(NexusTableModel.NexusHighlightingChangedEvent e);
        public void selectionChanged(NexusTableModel.NexusSelectionChangedEvent e);
    }

    List<JFreeChartScatterPlot.ScatterPlotListener> listeners = new ArrayList<>();

    public void addScatterPlotListener(JFreeChartScatterPlot.ScatterPlotListener li) {
        this.listeners.add(li);
    }
    public void removeScatterPlotListener(JFreeChartScatterPlot.ScatterPlotListener li) {
        this.listeners.remove(li);
    }

    private void fireHighlightingChangedEvent(NexusTableModel.NexusHighlightingChangedEvent event) {
        for(JFreeChartScatterPlot.ScatterPlotListener li : listeners) {
            li.highlightingChanged(event);
        }
    }

    private XYChartCreator.CombinedKeyedXYZDataset<String> data;

    public String getKeyForIndex(int i) {
        return (String) this.data.getKey(i);
    }
    public int getIndexForKey(String key) {
        return this.data.getIndex(key);
    }

    private JFreeChart chart;

    public JFreeChart getChart() {
        if(this.chart==null) {
            reinitPlot();
        }
        return this.chart;
    }

    public XYPlot getXYPlot() {
        return this.chart.getXYPlot();
    }

    public void reinitPlot() {
        KeyedValuesDataset px = this.dataX;//new DefaultKeyedValuesDataset();
        KeyedValuesDataset py = this.dataY;//new DefaultKeyedValuesDataset();
        KeyedValuesDataset cc = this.dataCol;//new DefaultKeyedValuesDataset();

        if(cc==null) {
            Random rand = new Random();
            DefaultKeyedValuesDataset ccn = new DefaultKeyedValuesDataset();
            //for(String ri : this.ntm.getAllRows()) { ccn.setValue(ri,rand.nextDouble());}
            for(Object ri : px.getKeys()) { ccn.setValue((String)ri,rand.nextDouble());}
            cc = ccn;
            this.dataCol = ccn;
        }

        this.data = new XYChartCreator.CombinedKeyedXYZDataset<>(px, py, cc);

        //JFreeChart chart = ChartFactory.createScatterPlot("test","x","y", data);
        //JFreeChart chart = ChartFactory.createScatterPlot(null,null,null, data);
        this.chart = ChartFactory.createScatterPlot(null,null,null, data);

        if(false) {
            chart.removeLegend();
            chart.getXYPlot().setBackgroundPaint(Color.black);
            chart.setBackgroundPaint(Color.black);
            chart.getXYPlot().setDomainGridlinesVisible(false);
            chart.getXYPlot().setRangeGridlinesVisible(false);
            chart.getXYPlot().getDomainAxis().setVisible(false);
            chart.getXYPlot().getRangeAxis().setVisible(false);
        }

        chart.setBorderVisible(true);
        chart.setBorderPaint(Color.orange.darker());
        chart.setBorderStroke(new BasicStroke(2));

        //((XYLineAndShapeRenderer)(chart.getXYPlot().getRenderer()).;
        //XYShapeRenderer renderer = new XYShapeRenderer();
        this.renderer = new XYShapeRenderer();
        renderer.setPaintScale(new ColorMapHelper.SpectrumPaintScale(0,1));

        chart.getXYPlot().setRenderer(renderer);
        //return chart;
    }

    public XYShapeRenderer getXYShapeRenderer() {
        return this.renderer;
    }

    public void setRendererShapeSize(int size) {
        Shape ci = new Ellipse2D.Double(-size/2,-size/2,size,size);
        this.renderer.setSeriesShape(0,ci);
    }

    /**
     *
     * @param classes
     * @param paints
     */
    public void setMultipleAnnotations(Map<String,List<Integer>> classes, Map<Integer,Paint> paints ) {
        multiClassAnnotations.stream().forEach( ai -> this.getXYPlot().removeAnnotation(ai) );
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
            this.getXYPlot().addAnnotation(hlan,true);
            //}
        }
    }



    public void setWithoutAxisAndLegend() {
        chart.removeLegend();
        chart.getXYPlot().setBackgroundPaint(Color.black);
        chart.setBackgroundPaint(Color.black);
        chart.getXYPlot().setDomainGridlinesVisible(false);
        chart.getXYPlot().setRangeGridlinesVisible(false);
        chart.getXYPlot().getDomainAxis().setVisible(false);
        chart.getXYPlot().getRangeAxis().setVisible(false);
    }




    public static class NumericalDatasourceKeyedDataset<U> implements KeyedValuesDataset {
        private NexusTableModel ntm;
        private NumericalDatasource<U> nds;

        public NumericalDatasourceKeyedDataset(NexusTableModel ntm, NumericalDatasource<U> nds) {
            this.ntm = ntm;
            this.nds = nds;
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
            if( nds.hasValue((String)comparable) ) {
                return nds.getValue((String) comparable);
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
            if( nds.hasValue(ri) ) {
                return nds.getValue(ri);
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


    public static class SetPointSizeAction extends AbstractAction {
        private List<ScatterPlotModel> fcs;
        private int size;
        public SetPointSizeAction(List<ScatterPlotModel> fcs,int size) {
            super("Set point size to "+String.format("%d",size));
            this.fcs = fcs;
            this.size = size;
        }
        @Override
        public void actionPerformed(ActionEvent e) {
            for(ScatterPlotModel fc2 : fcs){
                fc2.setRendererShapeSize((int) size);
            }
        }
    }

    public static class SetColorNumericalDatasource extends AbstractAction {
        private List<ScatterPlotModel> fcs;
        private NumericalDatasource nds;
        public SetColorNumericalDatasource(List<ScatterPlotModel> fcs,NumericalDatasource nds) {
            super(nds.getName());
            this.fcs = fcs;
            this.nds = nds;
        }
        @Override
        public void actionPerformed(ActionEvent e) {
            for(ScatterPlotModel fc2 : fcs){
                fc2.setColorValues(nds);
            }
        }
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



}
