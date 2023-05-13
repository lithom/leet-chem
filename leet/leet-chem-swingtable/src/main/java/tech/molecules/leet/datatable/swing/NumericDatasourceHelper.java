package tech.molecules.leet.datatable.swing;

import org.apache.commons.lang3.tuple.Pair;
import tech.molecules.leet.datatable.DataTable;
import tech.molecules.leet.datatable.DataTableColumn;
import tech.molecules.leet.datatable.NumericDatasource;
import tech.molecules.leet.datatable.filter.NumericRangeFilter;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.util.ArrayList;
import java.util.DoubleSummaryStatistics;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Collectors;

public class NumericDatasourceHelper {


    /**
     *
     * @param col
     * @param targetPanel this panel will be filled with the GUI for the filter
     * @return
     */
    public static List<Pair<String[], Action>> createFilterActions(DataTable dt, DataTableColumn<?,Double> col , Supplier<JPanel> targetPanel) {
        List<Pair<String[], Action>> list = col.getNumericDatasources().stream().flatMap( xi -> createFilterActions(dt,col,xi,targetPanel).stream() ).collect(Collectors.toList());
        return list;
    }

    public static List<Pair<String[], Action>> createFilterActions(DataTable dt, DataTableColumn<?,Double> col, NumericDatasource nd, Supplier<JPanel> targetPanel) {
        List<Pair<String[], Action>> list = new ArrayList<>();
        list.add(Pair.of(new String[]{nd.getName(),"Add Range Filter"},new AddNumericFilterAction(dt,col,nd,targetPanel)));
        return list;
    }

    public static DoubleSummaryStatistics computeSummaryStatistics(DataTable t, DataTableColumn c, NumericDatasource nd) {
        return t.getAllKeys().parallelStream().map( xi -> c.getRawValue(xi) ).mapToDouble( vi -> (double) nd.evaluate(vi) ).summaryStatistics();
    }

    public static class AddNumericFilterAction extends AbstractAction {
        DataTable dtable;
        DataTableColumn<?,Double> col;
        NumericDatasource nd;
        Supplier<JPanel> targetPanel;

        public AddNumericFilterAction(DataTable dt, DataTableColumn<?, Double> col, NumericDatasource nd, Supplier<JPanel> targetPanel) {
            this.dtable = dt;
            this.col = col;
            this.nd = nd;
            this.targetPanel = targetPanel;
        }

        @Override
        public void actionPerformed(ActionEvent e) {
            NumericRangeFilter nrf = new NumericRangeFilter();
            NumericRangeFilterController fc = new NumericRangeFilterController(dtable,col,nd,nrf);
            JPanel pi = targetPanel.get();
            pi.removeAll();
            pi.setLayout(new BorderLayout());
            pi.add(pi,BorderLayout.CENTER);
        }
    }

}
