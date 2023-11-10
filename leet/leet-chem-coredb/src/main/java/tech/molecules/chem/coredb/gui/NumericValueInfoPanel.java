package tech.molecules.chem.coredb.gui;

import tech.molecules.chem.coredb.aggregation.NumericValueInfo;

import javax.swing.*;

public class NumericValueInfoPanel extends JPanel {

    private NumericValueInfo numericValueInfo;

    public NumericValueInfoPanel() {
        this.reinit();
    }

    public void setNumericValueInfo(NumericValueInfo numericValueInfo) {
        this.numericValueInfo = numericValueInfo;
        this.reinit();
    }

    private void reinit() {

    }

}
