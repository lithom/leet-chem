package tech.molecules.leet.similarity;

import com.actelion.research.chem.StereoMolecule;
import com.formdev.flatlaf.FlatLightLaf;
import org.apache.commons.lang3.tuple.Pair;
import org.jfree.data.general.DefaultKeyedValuesDataset;
import org.knowm.xchart.XChartPanel;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.XYSeries;
import tagbio.umap.Umap;
import tech.molecules.leet.chem.ChemUtils;
import tech.molecules.leet.clustering.action.CreateClusteringToolAction;
import tech.molecules.leet.table.*;
import tech.molecules.leet.table.chart.JFreeChartScatterPlot2;
import tech.molecules.leet.table.chart.ScatterPlotModel;
import tech.molecules.leet.table.chart.XYChartCreator;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

public class UMapHelper {

    public static class UMapXYChartConfig {
        public final NStructureDataProvider dataset;
        public final NSimilarityColumn xCol;

        public UMapXYChartConfig(NStructureDataProvider dataset, NSimilarityColumn xCol) {
            this.dataset = dataset;
            this.xCol = xCol;
        }
    }

    public static JPanel createChart(NexusTableModel ntm, UMapXYChartConfig conf) {
        double[][] Sim = XYChartCreator.evaluateSimilarityDS(conf.dataset,conf.xCol,ntm.getVisibleRows());//ntm.getAllRows());
        //double[][] umap = computeUMap(ntm.getAllRows().toArray(new String[0]),Sim);
        double[][] umap = computeUMap(ntm.getVisibleRows().toArray(new String[0]),Sim);

        double umapx[] = new double[umap.length];
        double umapy[] = new double[umap.length];
        for(int zi=0;zi<umapx.length;zi++) {
            umapx[zi] = umap[zi][0];
            umapy[zi] = umap[zi][1];
        }

        XYChart chart = new XYChartBuilder().width(600).height(600).title("test")
                .xAxisTitle("umap1").yAxisTitle("umap2").build();
        chart.getStyler().setDefaultSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Scatter);
        chart.addSeries("umap",umapx,umapy);

        //new SwingWrapper<>(chart).displayChart();
        //ChartPanel cp = new ChartPanel(chart);
        XChartPanel<XYChart> panel = new XChartPanel<>(chart);
        return panel;
    }


    public static ScatterPlotModel createChart2(NexusTableModel ntm, UMapXYChartConfig conf) {
        //double[][] Sim = XYChartCreator.evaluateSimilarityDS(conf.dataset,conf.xCol,ntm.getAllRows());
        //double[][] umap = UMapHelper.computeUMap(ntm.getAllRows().toArray(new String[0]),Sim);
        double[][] Sim = XYChartCreator.evaluateSimilarityDS(conf.dataset,conf.xCol,ntm.getVisibleRows());
        double[][] umap = UMapHelper.computeUMap(ntm.getVisibleRows().toArray(new String[0]),Sim);


        double umapx[] = new double[umap.length];
        double umapy[] = new double[umap.length];
        for(int zi=0;zi<umapx.length;zi++) {
            umapx[zi] = umap[zi][0];
            umapy[zi] = umap[zi][1];
        }

        List<String> keys_a = new ArrayList<>();
        DefaultKeyedValuesDataset data_x = new DefaultKeyedValuesDataset();
        DefaultKeyedValuesDataset data_y = new DefaultKeyedValuesDataset();
        DefaultKeyedValuesDataset data_z = new DefaultKeyedValuesDataset();

        Random r = new Random();
        for(int zi=0;zi<ntm.getVisibleRows().size();zi++) {
            //keys_a.add(ntm.getAllRows().get(zi));
            keys_a.add(ntm.getVisibleRows().get(zi));
            data_x.setValue(keys_a.get(zi),umapx[zi]);
            data_y.setValue(keys_a.get(zi),umapy[zi]);
            //data_z.setValue(keys_a.get(zi), data_x.getValue(keys_a.get(zi)).doubleValue() + data_y.getValue(keys_a.get(zi)).doubleValue() );
        }

        //XYChartCreator.CombinedKeyedXYDataset<String> data  = new XYChartCreator.CombinedKeyedXYDataset<>(data_x,data_y);
        //XYChartCreator.CombinedKeyedXYZDataset<String> data2 = new XYChartCreator.CombinedKeyedXYZDataset<>(data_x,data_y,data_z);

        ScatterPlotModel spm = new ScatterPlotModel(ntm,data_x,data_y);
        //JFreeChartScatterPlot2 plot = new JFreeChartScatterPlot2(spm);
        //plot.setColor(data_z);

        return spm;

        //JFreeChart chart = ChartFactory.createScatterPlot("test","x","y", data);
        //((XYLineAndShapeRenderer)(chart.getXYPlot().getRenderer()).;
        //XYShapeRenderer renderer = new XYShapeRenderer();
        //renderer.setPaintScale(new GrayPaintScale());
        //chart.getXYPlot().setRenderer(renderer);
        //ChartPanel cp = new ChartPanel(chart,true,true,true,false,true);



//        CrosshairOverlay crosshairOverlay=new CrosshairOverlay();
//        float[] dash={2f,0f,2f};
//        BasicStroke bs=new BasicStroke(1,BasicStroke.CAP_BUTT,BasicStroke.JOIN_ROUND,1.0f,dash,2f);
//        Crosshair xCrosshair=new Crosshair(Double.NaN,Color.GRAY,bs);
//        xCrosshair.setLabelBackgroundPaint(new Color(0f,0f,0f,1f));
//        xCrosshair.setLabelFont(xCrosshair.getLabelFont().deriveFont(14f));
//        xCrosshair.setLabelPaint(new Color(1f,1f,1f,1f));
//        //xCrosshair.setLabelVisible(true);
//        Crosshair yCrosshair=new Crosshair(Double.NaN,Color.GRAY,bs);
//        yCrosshair.setLabelBackgroundPaint(new Color(0f,0f,0f,1f));
//        yCrosshair.setLabelFont(xCrosshair.getLabelFont().deriveFont(14f));
//        yCrosshair.setLabelPaint(new Color(1f,1f,1f,1f));
//        //yCrosshair.setLabelVisible(true);
//        xCrosshair.setVisible(true);
//        yCrosshair.setVisible(true);
//        crosshairOverlay.addDomainCrosshair(xCrosshair);
//        crosshairOverlay.addRangeCrosshair(yCrosshair);
//        cp.addOverlay(crosshairOverlay);
//
//
//
//        return cp;
    }


    public static double[][] computeUMap(String[] data, double dist[][]) {
        //haifeng smile stuff somehow tricky to make it work, due to cppbla openblas arpack stuff..
        //UMAP umap = UMAP.of( data, new ScaffoldFinder.ArrayDistance<>(Arrays.asList(data),dist));
        //double umap_coords[][] = umap.coordinates;
        Umap umap = new Umap();
        umap.setNumberComponents(2);         // number of dimensions in result
        umap.setNumberNearestNeighbours(8);
        umap.setThreads(4);                  // use > 1 to enable parallelism
        final double[][] result = umap.fitTransform(dist);
        return result;
    }



    public static void main(String args[]) {

        FlatLightLaf.setup();
        try {
            UIManager.setLookAndFeel( new FlatLightLaf() );
        } catch( Exception ex ) {
            System.err.println( "Failed to initialize LaF" );
        }

        String fa = "C:\\datasets\\hitexp_examples\\orexin\\structures_a.txt";
        List<StereoMolecule> mols = new ArrayList<>();

        try(BufferedReader in = new BufferedReader(new FileReader(fa))) {
            String line = null;
            while( (line=in.readLine()) != null ) {
                mols.add(ChemUtils.parseIDCode(line));
            }
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        NexusTableModel ntm = new NexusTableModel();
        List<String[]> mols_a = new ArrayList<>();
        for(int zi=0;zi<mols.size();zi++) { mols_a.add(new String[]{"Mol-"+String.format("%04d",zi),mols.get(zi).getIDCode()}); }
        DefaultStructureDataProvider sdp = new DefaultStructureDataProvider(mols_a);
        ntm.addNexusColumn(sdp,new StructureColumn(true));

        NexusTable nt = new NexusTable(ntm);
        JScrollPane jsp = new JScrollPane(nt);
        ntm.updateFiltering();
        ntm.setAllRows(mols_a.stream().map(si -> si[0] ).collect(Collectors.toList()));

        JFrame fi = new JFrame();
        fi.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);

        fi.getContentPane().setLayout(new BorderLayout());
        fi.getContentPane().add(jsp);

        fi.setSize(800,600);
        fi.setVisible(true);

        nt.setRowHeight(100);

        CreateClusteringToolAction ccta = new CreateClusteringToolAction("Clustering",ntm,sdp,null);
        ccta.actionPerformed(new ActionEvent(fi,13,"GO!"));

    }


}
