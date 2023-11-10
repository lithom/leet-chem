package tech.molecules.leet.datatable.microleet.task;

import tech.molecules.leet.datatable.DataTable;
import tech.molecules.leet.datatable.chart.jfc.VisualizationComponent;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.util.function.Supplier;

public class AddPlotAction extends AbstractAction {

    private DataTable table;
    private Supplier<JPanel> panelSupplier;

    public AddPlotAction(DataTable table, Supplier<JPanel> panelSupplier) {
        super("Add Plot");
        this.table = table;
        this.panelSupplier = panelSupplier;
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        VisualizationComponent vc = new VisualizationComponent(table);
        JPanel pi = panelSupplier.get();
        pi.removeAll();
        pi.setLayout(new BorderLayout());
        pi.add(vc,BorderLayout.CENTER);
        pi.revalidate();
        pi.repaint();
    }
}
