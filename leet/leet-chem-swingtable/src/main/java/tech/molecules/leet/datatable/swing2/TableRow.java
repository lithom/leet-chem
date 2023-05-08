package tech.molecules.leet.datatable.swing2;

import javax.swing.*;

public class TableRow extends JPanel {

    private InteractiveTable table;

    public class TableCell extends JPanel {
        private int col;
        public TableCell(int col) {
            super(null, true);
            this.col = col;
        }
        public void initData(Object o) {

        }
    }

}
