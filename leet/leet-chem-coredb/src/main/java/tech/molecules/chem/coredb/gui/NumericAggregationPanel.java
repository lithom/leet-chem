package tech.molecules.chem.coredb.gui;

import com.actelion.research.gui.VerticalFlowLayout;
import tech.molecules.chem.coredb.aggregation.NumericAggregationInfo;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;
import java.util.List;

public class NumericAggregationPanel extends JPanel {

    private NumericAggregationInfo numericAggregationInfo;
    private List<String> valueOptions = new ArrayList<>();

    private JPanel panelLeft;
    private JTextField textfieldValue;
    private JTextField textfieldMethod;
    private JCheckBox  checkboxLogarithmic;

    private JPanel panelRight;

    private JComboBox<String> comboBoxValues;
    private JComboBox<String> comboBoxMethods;


    public NumericAggregationPanel() {
    }

    public void setNumericAggregationInfo(NumericAggregationInfo numericAggregationInfo) {
        this.numericAggregationInfo = numericAggregationInfo;
        this.reinit();
    }

    public void setValueOptions(List<String> values) {
        this.valueOptions = values;
        this.reinit();
    }

    private void reinit() {
        this.removeAll();
        this.setLayout(new FlowLayout(FlowLayout.LEFT));

        this.panelLeft = new JPanel();
        this.panelLeft.setLayout(new VerticalFlowLayout(VerticalFlowLayout.LEFT,VerticalFlowLayout.TOP));
        this.textfieldValue = new JTextField(16);


        this.panelRight = new JPanel();
        this.textfieldMethod = new JTextField(10);

        this.panelRight.setLayout(new VerticalFlowLayout(VerticalFlowLayout.LEFT,VerticalFlowLayout.TOP));

    }



}
