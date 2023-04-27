package tech.molecules.leet.datatable.swing;

import tech.molecules.leet.datatable.DataTable;

import javax.swing.*;
import javax.swing.table.DefaultTableCellRenderer;
import java.awt.*;

public class DefaultSwingTable extends JPanel {

    private DefaultSwingTableModel model;

    private JTable table = new JTable();
    private JScrollPane scrollPane = new JScrollPane(table);


    public DefaultSwingTable(DefaultSwingTableModel model) {
        this.model = model;
        reinitGUI();
    }

    private void reinitGUI() {
        this.removeAll();
        this.setLayout(new BorderLayout());

        this.scrollPane = new JScrollPane();
        this.table      = new JTable(  );
    }


    public class DataTableCellRenderer extends DefaultTableCellRenderer {

        @Override
        public Component getTableCellRendererComponent(JTable table, Object value, boolean isSelected,
                                                       boolean hasFocus, int row, int column) {
            Component component = super.getTableCellRendererComponent(table, value, isSelected, hasFocus, row, column);

            // Retrieve the CellState for the current cell
            DataTable.CellState cellState = model.getCellState(row, column); // Implement this method to get the CellState from your data model

            // Create a JPanel with the specified background color
            JPanel panel = new JPanel();
            panel.setBackground(cellState.backgroundColor);

            // Create a custom border with the selection colors
            panel.setBorder(new MultiColorBorder(cellState.selectionColors, 40));

            // Add the original component (from the specific renderer) to the panel
            panel.add((JComponent) component);

            return panel;
        }
    }
}
