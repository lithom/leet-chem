package tech.molecules.leet.table.chart;

import com.formdev.flatlaf.FlatLightLaf;
import org.jfree.chart.*;
import org.jfree.chart.entity.ChartEntity;
import org.jfree.chart.entity.XYItemEntity;
import org.jfree.chart.renderer.GrayPaintScale;
import org.jfree.chart.renderer.xy.XYShapeRenderer;
import org.jfree.data.general.DefaultKeyedValuesDataset;
import org.jfree.data.general.KeyedValuesDataset;
import org.jfree.data.xy.AbstractXYDataset;
import org.jfree.data.xy.AbstractXYZDataset;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.XYSeries;
import tech.molecules.leet.table.NColumn;
import tech.molecules.leet.table.NDataProvider;
import tech.molecules.leet.table.NSimilarityColumn;
import tech.molecules.leet.table.NexusTableModel;

import javax.swing.*;
import java.awt.*;
import java.util.List;
import java.util.*;


public class XYChartCreator {

    public static class XYChartConfig {
        public final NDataProvider dataset;
        public final NColumn xCol;
        public final String  xDS;
        public final NColumn yCol;
        public final String  yDS;

        public XYChartConfig(NDataProvider dataset, NColumn xCol, String xDS, NColumn yCol, String yDS) {
            this.dataset = dataset;
            this.xCol = xCol;
            this.xDS = xDS;
            this.yCol = yCol;
            this.yDS = yDS;
        }
    }

//    public static class FixedXYChartConfig {
//        String labelX, labelY;
//        List<String> rows;
//        double[][]   xy;
//        public XYChartConfig(NDataset dataset, NColumn xCol, String xDS, NColumn yCol, String yDS) {
//            this.dataset = dataset;
//            this.xCol = xCol;
//            this.xDS = xDS;
//            this.yCol = yCol;
//            this.yDS = yDS;
//        }
//    }

    // replace by createChart method for creating charts with given xy input?
//    public static void createChart(NexusTableModel ntm, UMapXYChartConfig conf) {
//        double[][] Sim = evaluateSimilarityDS(conf.dataset,conf.xCol,ntm.getAllRows());
//        double[][] umap = SimilarityHelper.computeUMap(ntm.getAllRows().toArray(new String[0]),Sim);
//
//        double umapx[] = new double[umap.length];
//        double umapy[] = new double[umap.length];
//        for(int zi=0;zi<umapx.length;zi++) {
//            umapx[zi] = umap[zi][0];
//            umapy[zi] = umap[zi][1];
//        }
//
//        XYChart chart = new XYChartBuilder().width(600).height(600).title("test")
//                .xAxisTitle("umap1").yAxisTitle("umap2").build();
//        chart.getStyler().setDefaultSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Scatter);
//
//        new SwingWrapper<>(chart).displayChart();
//    }

    public static void createChart(NexusTableModel ntm, XYChartConfig conf) {
        String lx = conf.xCol.getName()+":"+conf.xDS;
        String ly = conf.yCol.getName()+":"+conf.yDS;
        XYChart chart = new XYChartBuilder().width(600).height(600).title("test")
                .xAxisTitle(lx).yAxisTitle(ly).build();
        chart.getStyler().setDefaultSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Scatter);

        double ddx[] = evaluateNumericalDS(conf.dataset,conf.xCol,conf.xDS,ntm.getAllRows());
        double ddy[] = evaluateNumericalDS(conf.dataset,conf.yCol,conf.yDS,ntm.getAllRows());

        chart.addSeries("all",ddx,ddy);
        new SwingWrapper<>(chart).displayChart();
    }

    //public static void createChart2(NexusTable nt, XYChartConfig conf) {
    public static void createChart2() {
//        String lx = conf.xCol.getName()+":"+conf.xDS;
//        String ly = conf.yCol.getName()+":"+conf.yDS;
//        //XYData
//        double ddx[] = evaluateNumericalDS(conf.dataset,conf.xCol,conf.xDS,nt.getTableModel().getAllRows());
//        double ddy[] = evaluateNumericalDS(conf.dataset,conf.yCol,conf.yDS,nt.getTableModel().getAllRows());

        List<String> keys_a = new ArrayList<>();
        DefaultKeyedValuesDataset data_x = new DefaultKeyedValuesDataset();
        DefaultKeyedValuesDataset data_y = new DefaultKeyedValuesDataset();
        DefaultKeyedValuesDataset data_z = new DefaultKeyedValuesDataset();

        Random r = new Random();
        for(int zi=0;zi<4000;zi++) {
            keys_a.add("data_"+zi);
            data_x.setValue(keys_a.get(zi),r.nextDouble());
            data_y.setValue(keys_a.get(zi),r.nextDouble());
            data_z.setValue(keys_a.get(zi), data_x.getValue(keys_a.get(zi)).doubleValue() + data_y.getValue(keys_a.get(zi)).doubleValue() );
        }




        //DefaultXYDataset xyd = new DefaultXYDataset();
        CombinedKeyedXYDataset<String>  data  = new CombinedKeyedXYDataset<>(data_x,data_y);
        CombinedKeyedXYZDataset<String> data2 = new CombinedKeyedXYZDataset<>(data_x,data_y,data_z);

        JFreeChart chart = ChartFactory.createScatterPlot("test","x","y", data2);
        //((XYLineAndShapeRenderer)(chart.getXYPlot().getRenderer()).;
        XYShapeRenderer renderer = new XYShapeRenderer();
        renderer.setPaintScale(new GrayPaintScale());
        chart.getXYPlot().setRenderer(renderer);

        chart.getXYPlot().getChart().getXYPlot().setDomainCrosshairPaint(new Color(100,100,100));
        chart.getXYPlot().getChart().getXYPlot().setDomainCrosshairStroke(new BasicStroke(1));
        chart.getXYPlot().getChart().getXYPlot().setDomainCrosshairVisible(true);
        chart.getXYPlot().getChart().getXYPlot().setRangeCrosshairPaint(new Color(100,100,100));
        chart.getXYPlot().getChart().getXYPlot().setRangeCrosshairStroke(new BasicStroke(1));
        chart.getXYPlot().getChart().getXYPlot().setRangeCrosshairVisible(true);

        //chart.getXYPlot().setRenderer();
        //JFreeChart chart = ChartFactory.create

        ChartPanel cp = new ChartPanel(chart);
        cp.addChartMouseListener(new ChartMouseListener() {
            @Override
            public void chartMouseClicked(ChartMouseEvent chartMouseEvent) {

            }

            @Override
            public void chartMouseMoved(ChartMouseEvent chartMouseEvent) {
                ChartEntity ent = chartMouseEvent.getEntity();
                if( ent != null) {
                    if( ent instanceof XYItemEntity) {
                        System.out.println("Entity: " + ent.toString() );
                    }
                }
            }
        });

        JFrame fi = new JFrame();
        fi.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        fi.getContentPane().setLayout(new BorderLayout());
        fi.getContentPane().add(cp,BorderLayout.CENTER);
        fi.setSize(400,400);
        fi.setVisible(true);
        //chart.addSeries("all",ddx,ddy);
        //new SwingWrapper<>(chart).displayChart();
    }

    public static void main(String args[]) {

        FlatLightLaf.setup();
        try {
            UIManager.setLookAndFeel( new FlatLightLaf() );
        } catch( Exception ex ) {
            System.err.println( "Failed to initialize LaF" );
        }

        createChart2();
    }

    public static class CombinedKeyedXYDataset<T extends Comparable> extends AbstractXYDataset {

        List<T> keys = new ArrayList<>();
        //Map<T,Double> data_x;
        //Map<T,Double> data_y;
        KeyedValuesDataset dx;
        KeyedValuesDataset dy;

        public CombinedKeyedXYDataset(KeyedValuesDataset x, KeyedValuesDataset y) {
            this.setKeyedXY(x,y);
        }

        public void setKeyedXY(KeyedValuesDataset x, KeyedValuesDataset y) {
            SortedSet all_keys = new TreeSet<>();
            all_keys.addAll( x.getKeys() );
            all_keys.addAll( y.getKeys() );
            keys = new ArrayList<>( all_keys );
            this.dx = x;
            this.dy = y;
            //for(T ki : keys) {
            //    x.get(ki)
            //}
        }

        @Override
        public int getSeriesCount() {
            return 1;
        }

        @Override
        public Comparable getSeriesKey(int i) {
            return 1;
        }

        @Override
        public int getItemCount(int i) {
            return this.keys.size();
        }

        @Override
        public Number getX(int i, int i1) {
            return dx.getValue( keys.get(i1) );
        }

        @Override
        public Number getY(int i, int i1) {
            return dy.getValue( keys.get(i1) );
        }
    }

    /**
     *
     *
     * @param <T>
     */
    public static class CombinedKeyedXYZDataset<T extends Comparable> extends AbstractXYZDataset implements KeyedValuesDataset {

        private List<T> keys = new ArrayList<>();
        //Map<T,Double> data_x;
        //Map<T,Double> data_y;
        private KeyedValuesDataset dx;
        private KeyedValuesDataset dy;
        private KeyedValuesDataset dz;

        private Map<T,Integer> indeces;

        public CombinedKeyedXYZDataset(KeyedValuesDataset x, KeyedValuesDataset y, KeyedValuesDataset z) {
            this.setKeyedXYZ(x,y,z);
        }

        public void setZ(KeyedValuesDataset zn) {
            this.dz = zn;
            this.fireDatasetChanged();
        }

//        public void fireDatasetChangedExplicitly() {
//            this.fireDatasetChanged();
//        }

        public void setKeyedXYZ(KeyedValuesDataset x, KeyedValuesDataset y, KeyedValuesDataset z) {
            SortedSet all_keys = new TreeSet<>();
            all_keys.addAll( x.getKeys() );
            all_keys.addAll( y.getKeys() );
            all_keys.addAll( z.getKeys() );
            keys = new ArrayList<>( all_keys );
            Map<T,Integer> mapToIndeces = new HashMap();
            for(int zi=0;zi<keys.size();zi++) { mapToIndeces.put(keys.get(zi),zi);}
            this.indeces = mapToIndeces;

            this.dx = x;
            this.dy = y;
            this.dz = z;
            //for(T ki : keys) {
            //    x.get(ki)
            //}
        }

        @Override
        public int getSeriesCount() {
            return 1;
        }

        @Override
        public Comparable getSeriesKey(int i) {
            return 1;
        }

        @Override
        public int getItemCount(int i) {
            return this.keys.size();
        }

        @Override
        public Number getX(int i, int i1) {
            return dx.getValue( keys.get(i1) );
        }

        @Override
        public Number getY(int i, int i1) {
            return dy.getValue( keys.get(i1) );
        }

        @Override
        public Number getZ(int i, int i1) {
            return dz.getValue( keys.get(i1) );
        }

        @Override
        public T getKey(int i) {
            return keys.get(i);
        }

        @Override
        public int getIndex(Comparable comparable) {
            return this.keys.indexOf(comparable);
        }

        @Override
        public List<T> getKeys() {
            return this.keys;
        }

        @Override
        public Number getValue(Comparable comparable) {
            return this.dx.getValue(comparable);
        }

        @Override
        public int getItemCount() {
            return this.keys.size();
        }

        @Override
        public Number getValue(int i) {
            return this.dx.getValue( this.keys.get(i) );
        }
    }

    public static double[] evaluateNumericalDS(NDataProvider nd, NColumn c, String ds, List<String> rids) {
        double results[] = new double[rids.size()];
        for(int zi=0;zi<rids.size();zi++) {
            results[zi] = c.evaluateNumericalDataSource(nd,ds,rids.get(zi));
        }
        return results;
    }

    public static double[][] evaluateSimilarityDS(NDataProvider nd, NSimilarityColumn c, List<String> rids) {
        double results[][] = new double[rids.size()][rids.size()];
        for(int zi=0;zi<rids.size()-1;zi++) {
            for(int zj=zi;zj<rids.size();zj++) {
                results[zi][zj] = c.evaluateValue(nd,rids.get(zi),rids.get(zj));
                results[zj][zi] = results[zi][zj];
            }
        }
        return results;
    }

}
