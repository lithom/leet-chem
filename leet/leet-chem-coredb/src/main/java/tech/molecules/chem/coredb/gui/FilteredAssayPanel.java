package tech.molecules.chem.coredb.gui;

import tech.molecules.chem.coredb.aggregation.FilteredAssayInfo;
import tech.molecules.chem.coredb.aggregation.NumericValueInfo;

import javax.swing.*;
import java.awt.*;

public class FilteredAssayPanel extends JPanel {

    private FilteredAssayInfo filteredAssayInfo;

    // GUI:

    private JPanel top;
    private JTextField textFieldAssay;
    private JButton buttonSetAssay;
    private JButton buttonSetFiltering;



    public FilteredAssayPanel() {
        reinit();
    }

    public void setData(FilteredAssayInfo info) {
        this.filteredAssayInfo = info;
        this.reinit();
    }

    private void reinit() {
        this.removeAll();
        this.setLayout(new BorderLayout());
        this.top = new JPanel();
        top.setLayout(new FlowLayout(FlowLayout.LEFT));
        this.textFieldAssay = new JTextField(20);
        //this.textFieldParam = new JTextField(16);
        this.buttonSetAssay     = new JButton("Set Assay..");
        this.buttonSetFiltering = new JButton("Set Filtering..");
        this.top.add(this.textFieldAssay);
        //this.top.add(this.textFieldParam);
        // SET DATA:
        this.add(this.top,BorderLayout.CENTER);
    }

}
