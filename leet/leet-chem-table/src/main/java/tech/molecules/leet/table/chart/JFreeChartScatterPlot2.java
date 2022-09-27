package tech.molecules.leet.table.chart;

import org.jfree.chart.ChartMouseEvent;
import org.jfree.chart.ChartMouseListener;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.entity.ChartEntity;
import org.jfree.chart.entity.EntityCollection;
import org.jfree.chart.entity.XYItemEntity;
import tech.molecules.leet.table.NumericalDatasource;
import tech.molecules.leet.table.gui.JNumericalDataSourceSelector;
import tech.molecules.leet.util.ColorMapHelper;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.stream.Collectors;

public class JFreeChartScatterPlot2 extends JPanel {
    private ScatterPlotModel model;

    //private JPanel chart;
    private ChartPanel cp;

    private boolean initMenuBar;
    private JMenuBar jmb;

    public JFreeChartScatterPlot2(ScatterPlotModel mi) {
        this(mi,true);
    }

    public JFreeChartScatterPlot2(ScatterPlotModel mi, boolean createMenuBar) {
        this.model = mi;
        this.initMenuBar = createMenuBar;
        init();
    }

    public ChartPanel getChartPanel() {
        return this.cp;
    }

    private void init() {

        JFreeChart chart = model.getChart();
        this.cp = new ChartPanel(chart);

        this.jmb = new JMenuBar();
        //initMenu(jmb);


        this.removeAll();

        this.setLayout(new BorderLayout());
        this.add(cp,BorderLayout.CENTER);


        if(initMenuBar) {
            this.jmb = new JMenuBar();
            JScatterPlotConfigurationMenuBar.initMenuBar(jmb,model.getNexusTableModel(), Collections.singletonList(model));
            this.add(jmb,BorderLayout.NORTH);
        }


        this.cp.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseExited(MouseEvent e) {
                // clear the highlighting annotations, if we really leave the component, also
                // based on x/y coordinates.
                if(!contains(e.getX(),e.getY()))
                {
                    model.setHighlight(new HashSet(), true);
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
                //if(chartMouseEvent.getTrigger().getButton()==MouseEvent.BUTTON3) {
                //    isMouseContextMenuShown = true;
                //}
            }

            @Override
            public void chartMouseMoved(ChartMouseEvent chartMouseEvent) {
                ChartEntity ent = chartMouseEvent.getEntity();
                if( ent != null) {
                    if( ent instanceof XYItemEntity) {
                        System.out.println("Entity: " + ent.toString() );
                    }
                }

                int highlightNNearestNeighbors = model.getHighlightNNearestNeighbors();

                if(highlightNNearestNeighbors>=1) {
                    //processHighlightNNearestNeighbors(  );
                    EntityCollection all_e = cp.getChartRenderingInfo().getEntityCollection();
                    EntityCollection ec = ScatterPlotModel.getNNearestEntities(all_e,chartMouseEvent.getTrigger().getX(),chartMouseEvent.getTrigger().getY(),highlightNNearestNeighbors);
                    System.out.println("to highlight: ");
                    for(Object cei : ec.getEntities()) {
                        System.out.println("nearest: "+cei.toString());
                    }
                    java.util.List<String> h_keys = (List<String>) ec.getEntities().stream().map(ei -> model.getKeyForXYEntity( (XYItemEntity) ei)).collect(Collectors.toList());
                    model.setHighlight( new HashSet(h_keys) , true );
                }
            }
        });
    }

    public double[] getPositionOfKey(String ski) {
        int ki = this.getModel().getIndexForKey(ski);
        double kpx = this.cp.getChartRenderingInfo().getEntityCollection().getEntity( ki ).getArea().getBounds2D().getCenterX();
        double kpy = this.cp.getChartRenderingInfo().getEntityCollection().getEntity( ki ).getArea().getBounds2D().getCenterY();
        return new double[] { kpx, kpy };
    }

    public ScatterPlotModel getModel() {
        return this.model;
    }


//    private void initMenu(JMenuBar jmb) {
//        JMenu jmsize = new JMenu("Marker");
//        jmsize.add(new SetPointSizeAction(2));
//        jmsize.add(new SetPointSizeAction(4));
//        jmsize.add(new SetPointSizeAction(8));
//        jmsize.add(new SetPointSizeAction(16));
//        jmb.add(jmsize);
//        JMenu jmcolor = new JMenu("Color");
//        JNumericalDataSourceSelector jndss = new JNumericalDataSourceSelector(new JNumericalDataSourceSelector.NumericalDataSourceSelectorModel(model.getNexusTableModel()), JNumericalDataSourceSelector.SELECTOR_MODE.OnlyJMenu);
//        JMenu jnds = jndss.getMenu();
//        jnds.setText("From Numerical Datasource");
//        jmcolor.add(jnds);
//        jndss.addSelectionListener(new JNumericalDataSourceSelector.SelectionListener() {
//            @Override
//            public void selectionChanged() {
//                // set coloring according to values of nds
//                model.setColorValues( jndss.getModel().getSelectedDatasource());
//            }
//        });
//        jmb.add(jmcolor);
//    }





}
