package tech.molecules.leet.datatable.swing;

import tech.molecules.leet.datatable.DataFilterType;
import tech.molecules.leet.datatable.DataTable;
import tech.molecules.leet.datatable.DataTableColumn;
import tech.molecules.leet.datatable.NumericDatasource;
import tech.molecules.leet.datatable.filter.NumericRangeFilter;
import tech.molecules.leet.gui.JSlimRangeSlider;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.util.DoubleSummaryStatistics;
import java.util.function.Consumer;
import java.util.function.Supplier;

public class NumericRangeFilterController extends AbstractSwingFilterController<Double> {

    public static class NumericRangePanel extends JPanel {
        private JSlimRangeSlider slider;
        private Consumer<double[]> callback_range;
        public NumericRangePanel(double domain_ab[], double range_ab[], Consumer<double[]> cb_range) {
            this.callback_range = cb_range;
            reinit(domain_ab,range_ab);
        }
        public void reinit(double domain_ab[], double range_ab[]) {
            this.removeAll();
            this.setLayout(new BorderLayout());
            this.slider = new JSlimRangeSlider(domain_ab[0],domain_ab[1]);
            if(range_ab!=null) {
                this.slider.setRange(range_ab[0], range_ab[1]);
            }
            this.add(slider,BorderLayout.CENTER);
            this.slider.addChangeListener(new ChangeListener() {
                @Override
                public void stateChanged(ChangeEvent e) {
                    callback_range.accept(slider.getRange());
                }
            });
        }
    }

    public NumericRangeFilterController(DataTable dt, DataTableColumn<?, Double> column, NumericDatasource<?> nds, NumericRangeFilter filter) {
        super(dt, column, filter.getDataFilterType());
        this.filter = filter;

        DoubleSummaryStatistics dss = NumericDatasourceHelper.computeSummaryStatistics(dt,column,nds);
        double[] domain_ab  = new double[]{dss.getMin(),dss.getMax()};
        double[] range_ab   = filter.getRange();
        this.panel = new NumericRangePanel(domain_ab,range_ab,(xi) -> filter.setRange(xi));
    }

    private NumericRangeFilter filter = new NumericRangeFilter();
    private JPanel panel;

    @Override
    public JPanel getConfigurationPanel() {
        return this.panel;
    }
}
