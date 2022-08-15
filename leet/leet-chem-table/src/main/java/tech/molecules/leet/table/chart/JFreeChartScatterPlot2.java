package tech.molecules.leet.table.chart;

import org.jfree.chart.ChartMouseEvent;
import org.jfree.chart.ChartMouseListener;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.entity.ChartEntity;
import org.jfree.chart.entity.EntityCollection;
import org.jfree.chart.entity.XYItemEntity;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.HashSet;
import java.util.List;
import java.util.stream.Collectors;

public class JFreeChartScatterPlot2 extends JPanel {
    private ScatterPlotModel model;

    //private JPanel chart;
    private ChartPanel cp;

    public JFreeChartScatterPlot2(ScatterPlotModel mi) {
        this.model = mi;

        init();
    }

    private void init() {

        JFreeChart chart = model.getChart();
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
}
